"""Simulation control flow and runtime"""

import asyncio
import os
from argparse import Namespace
from collections import deque
from typing import Any, Callable

import numpy as np
from PIL import Image
from isaacgym import gymapi, gymtorch, gymutil
import torch
from torch import Tensor

import config as cfg
import maze
from model import ActorCritic
from train import PPG
from accel import OpAccelerator, SessionTensorSignature, SCALE_TIME
from utils import eval_line_of_sight, get_available_file_idx
from utils_torch import adjust_depth_range, apply_quat_rot, get_eulz_from_quat
from utils_train import CheckpointTracker, NAdamW, SoftConstLRScheduler


class Interface:
    """Gym.Viewer wrapper describing observer and actor control over the sim."""

    TORQUE_STATES: 'dict[float, list[float]]' = {
        np.nan: [0., 0., 0., 0.],
        np.arctan2(0, 1): [1., 1., 1., 1.],
        np.arctan2(0, -1): [-1., -1., -1., -1.],
        np.arctan2(1, 0): [-1., 1., -1., 1.],
        np.arctan2(-1, 0): [1., -1., 1., -1.],
        np.arctan2(1, 1): [0., 1., 0., 1.],
        np.arctan2(-1, 1): [1., 0., 1., 0.],
        np.arctan2(-1, -1): [0., -1., 0., -1.],
        np.arctan2(1, -1): [-1., 0., -1., 0.]}

    OFFSET_BOT_AHEAD = [[maze.CAM_OFFSET[0], 0., 0.]]
    OFFSET_BOT_ABOVE = [[0., 0., maze.CAM_OFFSET[2]]]

    OFFSET_3RD_AHEAD = [[-maze.BOT_WIDTH*2, 0., 0.]]
    OFFSET_3RD_ABOVE = [[0., 0., maze.WALL_HALFHEIGHT]]

    OBS_EVENTS = [
        (gymapi.KEY_ESCAPE, 'end_session'), (gymapi.KEY_L, 'lvl_reset'),
        (gymapi.KEY_H, 'print_help'), (gymapi.KEY_G, 'print_debug'),
        (gymapi.KEY_N, 'cycle_env'), (gymapi.KEY_B, 'cycle_bot'),
        (gymapi.KEY_V, 'cycle_view'), (gymapi.KEY_I, 'save_view')]

    ACT_EVENTS = [
        (gymapi.KEY_W, 'move_forw'), (gymapi.KEY_S, 'move_back'),
        (gymapi.KEY_A, 'move_left'), (gymapi.KEY_D, 'move_right'),
        (gymapi.KEY_SPACE, 'alt_move'), (gymapi.KEY_C, 'recolour')]

    OBS_KEYS = set(name for _, name in OBS_EVENTS)
    ACT_KEYS = set(name for _, name in ACT_EVENTS)

    MKBD_EVENTS = OBS_EVENTS + ACT_EVENTS
    MKBD_KEYS = OBS_KEYS | ACT_KEYS

    VIEW_BOT = 2
    VIEW_TOP = 1
    VIEW_3RD = 0

    def __init__(self, session: 'Session'):
        self.session = session
        self.gym = session.gym
        self.sim = session.sim

        # Init viewer
        self.viewer = self.gym.create_viewer(self.sim.handle, gymapi.CameraProperties())
        self.viewer_top_pos = gymapi.Vec3(0., self.sim.env_width * 0.75, self.sim.env_width * 0.75)
        self.viewer_top_target = gymapi.Vec3(0., 0., 0.)

        if self.viewer is None:
            raise Exception('Failed to create viewer.')

        # Setup input events for user interaction
        for key, name in self.MKBD_EVENTS:
            self.gym.subscribe_viewer_keyboard_event(self.viewer, key, name)

        self.view = self.VIEW_TOP
        self.env_idx = 0
        self.bot_idx = 0
        self.all_bot_idx = 0

        self.offset_bot_ahead = torch.tensor(self.OFFSET_BOT_AHEAD, dtype=torch.float32)
        self.offset_bot_above = torch.tensor(self.OFFSET_BOT_ABOVE, dtype=torch.float32)
        self.offset_3rd_ahead = torch.tensor(self.OFFSET_3RD_AHEAD, dtype=torch.float32)
        self.offset_3rd_above = torch.tensor(self.OFFSET_3RD_ABOVE, dtype=torch.float32)

        self.torque_states = {
            key: torch.tensor(val, dtype=torch.float32, device=session.device)
            for key, val in self.TORQUE_STATES.items()}

        self.key_vec = np.zeros(5)

    def update_target_indices(self, env_inc: int = 0, bot_inc: int = 0):
        self.env_idx = (self.env_idx + env_inc) % self.sim.n_envs
        self.bot_idx = (self.bot_idx + bot_inc) % self.sim.n_bots
        self.all_bot_idx = self.env_idx * self.sim.n_bots + self.bot_idx

    def update_top_view(self):
        if self.view != self.VIEW_TOP:
            return

        self.gym.viewer_camera_look_at(
            self.viewer,
            self.sim.envs[self.env_idx].handle,
            self.viewer_top_pos,
            self.viewer_top_target)

    def update_bot_view(self):
        if self.view == self.VIEW_TOP:
            return

        elif self.view == self.VIEW_BOT:
            offset_ahead = self.offset_bot_ahead
            offset_above = self.offset_bot_above

        else:
            offset_ahead = self.offset_3rd_ahead
            offset_above = self.offset_3rd_above

        self.gym.refresh_actor_root_state_tensor(self.sim.handle)

        bot_pos = self.session.env_bot_pos3[self.env_idx, self.bot_idx].cpu()
        bot_ori = self.session.env_bot_ori[None, self.env_idx, self.bot_idx].cpu()

        vec_ahead = apply_quat_rot(bot_ori, offset_ahead)[0]
        ref_ahead = apply_quat_rot(bot_ori, self.offset_bot_ahead)[0]

        pos_view = bot_pos + offset_above[0] + vec_ahead
        pos_target = pos_view + ref_ahead

        self.gym.viewer_camera_look_at(
            self.viewer,
            self.sim.envs[self.env_idx].handle,
            gymapi.Vec3(*pos_view.numpy()),
            gymapi.Vec3(*pos_target.numpy()))

    def get_torque_from_key_vec(self) -> Tensor:
        """Compute 4-wheel torques from keyboard press state."""

        mvmt_forw = self.key_vec[0:2].sum()
        mvmt_left = self.key_vec[2:4].sum()
        scale = (maze.MOT_MAX_TORQUE / 2.) if self.key_vec[4] else maze.MOT_MAX_TORQUE

        if not (mvmt_forw or mvmt_left):
            return self.torque_states[np.nan]

        return self.torque_states.get(np.arctan2(mvmt_left, mvmt_forw), self.torque_states[np.nan]) * scale

    def get_debug_info(self) -> str:
        if self.session.async_temp_result is None:
            return 'State not yet evaluated.\n'

        obs_vec, obs_aux = self.session.async_temp_result
        obs_vec = obs_vec[self.all_bot_idx].cpu().numpy()
        obs_aux = obs_aux[self.all_bot_idx].cpu().numpy()

        time_left = (self.sim.ep_duration - self.session.env_run_times[self.env_idx]).item()
        tput = self.session.get_throughput()
        bot_ori = self.session.env_bot_ori[None, self.env_idx, self.bot_idx].cpu()
        z_angle = get_eulz_from_quat(bot_ori).item() * 180. / np.pi

        return (
            '\nSESSION\n'
            f'Time to ep. end | {max(0., time_left): .2f}s\n'
            f'Avg. throughput | {tput: .2f} (per bot per min)\n'
            f'Ori. angle (z)  | {z_angle: .0f}\n\n'

            'GUIDE\n'
            f'Air direction   | Front: {obs_aux[0]: .2f} | Left: {obs_aux[1]: .2f}\n'
            f'A*  direction   | Front: {obs_aux[2]: .2f} | Left: {obs_aux[3]: .2f}\n'
            f'Air proximity   |        {obs_aux[4]: .2f}\n'
            f'A*  proximity   |        {obs_aux[5]: .2f}\n'
            f'Goal in sight   | {"TRUE" if obs_aux[6] else "FALSE"}\n\n'

            'AUXILIARY\n'
            f'Time at goal    | {obs_aux[7]: .2f}s\n'
            f'Time on task    | {obs_aux[8] / SCALE_TIME: .2f}s\n'
            f'Time to ep. end | {obs_aux[9] / SCALE_TIME: .2f}s\n'
            f'New/done tasks  | {obs_aux[10]: .0f}\n'
            f'Avg. throughput | {obs_aux[11]: .2f}\n\n'

            'OBSERVATION\n'
            f'Time at goal    | {obs_vec[0]: .2f}s\n'
            f'Avg. throughput | {obs_vec[1]: .2f} (per bot per 6s)\n'
            f'Act. torques    | {obs_vec[2]: .2f}, {obs_vec[3]: .2f}, {obs_vec[4]: .2f}, {obs_vec[5]: .2f}\n'
            f'DOF vel.        | {obs_vec[6]: .2f}, {obs_vec[7]: .2f}, {obs_vec[8]: .2f}, {obs_vec[9]: .2f}\n'
            f'IMU ang. vel.   | {obs_vec[10]: .2f}, {obs_vec[11]: .2f}, {obs_vec[12]: .2f}\n'
            f'IMU accel.      | {obs_vec[13]: .2f}, {obs_vec[14]: .2f}, {obs_vec[15]: .2f}\n'
            f'IMU magnet.     | {obs_vec[16]: .2f}, {obs_vec[17]: .2f}, {obs_vec[18]: .2f}\n'
            f'Act. RGB        | {obs_vec[19]: .2f}, {obs_vec[20]: .2f}, {obs_vec[21]: .2f}\n'
            f'RGB goal        | {obs_vec[22]: .2f}, {obs_vec[23]: .2f}, {obs_vec[24]: .2f}\n'
            f'RGB receiver F  | {obs_vec[25]: .2f}, {obs_vec[26]: .2f}, {obs_vec[27]: .2f}\n'
            f'RGB receiver L  | {obs_vec[28]: .2f}, {obs_vec[29]: .2f}, {obs_vec[30]: .2f}\n'
            f'RGB receiver R  | {obs_vec[31]: .2f}, {obs_vec[32]: .2f}, {obs_vec[33]: .2f}\n')

    def save_camera_images(self) -> 'tuple[str, str]':
        self.gym.render_all_camera_sensors(self.sim.handle)
        self.gym.start_access_image_tensors(self.sim.handle)

        rgb = self.session.img_rgb_list[self.all_bot_idx][..., :3].cpu().numpy()
        dep = 255. * adjust_depth_range(-self.session.img_dep_list[self.all_bot_idx]).cpu().numpy()

        self.gym.end_access_image_tensors(self.sim.handle)

        # RGB
        file_idx = get_available_file_idx(cfg.DATA_DIR, 'img_rgbcam')
        filename_rgb = os.path.join(cfg.DATA_DIR, f'img_rgbcam_{file_idx:02d}.png')

        Image.fromarray(rgb.astype(np.uint8), mode='RGB').save(filename_rgb)

        # Depth
        file_idx = get_available_file_idx(cfg.DATA_DIR, 'img_depcam')
        filename_dep = os.path.join(cfg.DATA_DIR, f'img_depcam_{file_idx:02d}.png')

        Image.fromarray(dep.astype(np.uint8), mode='L').save(filename_dep)

        return filename_rgb, filename_dep

    def save_viewer_image(self) -> str:
        file_idx = get_available_file_idx(cfg.DATA_DIR, 'img_viewer')
        filename = os.path.join(cfg.DATA_DIR, f'img_viewer_{file_idx:02d}.png')

        self.gym.write_viewer_image_to_file(self.viewer, filename)

        return filename

    def eval_events(self):
        """Check for keyboard events, update torque, colours, and view."""

        if self.gym.query_viewer_has_closed(self.viewer):
            raise KeyboardInterrupt

        for event in self.gym.query_viewer_action_events(self.viewer):
            cmd_key, cmd_press = event.action, event.value > 0

            if cmd_key in self.OBS_KEYS:
                if not cmd_press:
                    continue

                if cmd_key == 'end_session':
                    raise KeyboardInterrupt

                elif cmd_key == 'lvl_reset':
                    self.session.reset_all()

                    print('All environments flagged for immediate reset.')

                elif cmd_key == 'print_help':
                    print('Available keys:')

                    for key_map in sorted(self.MKBD_EVENTS, key=lambda k: k[1]):
                        print(key_map)

                elif cmd_key == 'print_debug':
                    print(self.get_debug_info())

                # NOTE: Same as below
                elif cmd_key == 'cycle_env':
                    self.update_target_indices(env_inc=-1 if self.key_vec[-1] else 1)

                    print(f'Env/agent index switched to {self.env_idx}/{self.all_bot_idx}.')

                # NOTE: Previous bot torques are not automatically reset to zero
                elif cmd_key == 'cycle_bot':
                    self.update_target_indices(bot_inc=-1 if self.key_vec[-1] else 1)

                    print(f'Bot/agent index switched to {self.bot_idx}/{self.all_bot_idx}.')

                elif cmd_key == 'cycle_view':
                    self.view = (self.view + (-1 if self.key_vec[-1] else 1)) % 3
                    self.update_top_view()

                elif cmd_key == 'save_view':
                    if self.view == self.VIEW_BOT:
                        filename_rgb, filename_dep = self.save_camera_images()

                        print(
                            f'Saved camera RGB image to: {filename_rgb}\n'
                            f'Saved camera DEP image to: {filename_dep}\n')

                    else:
                        print(f'Saved viewer image to: {self.save_viewer_image()}')

            elif cmd_key in self.ACT_KEYS:
                if cmd_key == 'alt_move':
                    self.key_vec[-1] = event.value

                elif self.session.ctrl_mode != Session.CTRL_MAN:
                    if cmd_press:
                        print('Cannot command agents without manual control mode.')

                    continue

                elif cmd_key == 'recolour':
                    if cmd_press:
                        self.session.actions[:, -cfg.RGB_VEC_SIZE:] = \
                            torch.rand((self.sim.n_all_bots, cfg.RGB_VEC_SIZE), device=self.session.device)

                    continue

                elif cmd_key == 'move_forw':
                    self.key_vec[0] = event.value

                elif cmd_key == 'move_back':
                    self.key_vec[1] = -event.value

                elif cmd_key == 'move_left':
                    self.key_vec[2] = event.value

                elif cmd_key == 'move_right':
                    self.key_vec[3] = -event.value

                self.session.actions[self.all_bot_idx, :cfg.DOF_VEC_SIZE] = self.get_torque_from_key_vec()

    def sync_redraw(self):
        """Draw the scene in the viewer, syncing sim with real-time."""

        self.update_bot_view()
        self.gym.draw_viewer(self.viewer, self.session.sim.handle, False)
        self.gym.sync_frame_time(self.session.sim.handle)

    def reset(self):
        self.key_vec.fill(0)
        self.update_bot_view()


class Session(SessionTensorSignature):
    """
    The main process descriptor connecting environment, operational,
    and training constructs, describing the flow of the sim on a tensor level,
    and providing several interfacing options.
    """

    CTRL_AI = 3
    CTRL_RL = 2
    CTRL_GEN = 1
    CTRL_MAN = 0

    REC_ALL = 2
    REC_VEC = 1
    REC_NONE = 0

    MDL_FULL = 3
    MDL_GUIDE = 2
    MDL_COM = 1
    MDL_BASE = 0

    MODEL_OPTIONS = {
        MDL_BASE: {
            'suffix': 'base',
            'com': False,
            'guide': False},
        MDL_COM: {
            'suffix': 'com',
            'com': True,
            'guide': False},
        MDL_GUIDE: {
            'suffix': 'guide',
            'com': False,
            'guide': True},
        MDL_FULL: {
            'suffix': 'full',
            'com': True,
            'guide': True}}

    NULL_INFO = {}

    goal_pos_arr: np.ndarray
    goal_wallgrid_idx: np.ndarray

    env_bot_pos3: Tensor
    env_bot_pos: Tensor
    env_bot_ori: Tensor

    ARGS = [
        {'name': '--level', 'type': int, 'default': 4, 'help': 'Maze complexity level.'},
        {'name': '--keep_level', 'type': int, 'default': 4, 'help': 'Min. level that keeps its preset structure.'},
        {'name': '--regen', 'type': int, 'default': 0, 'help': 'Option to fully regenerate environments on reset.'},
        {'name': '--n_bots', 'type': int, 'default': -1, 'help': 'Number of agents per environment.'},
        {'name': '--n_envs', 'type': int, 'default': -1, 'help': 'Number of parallel environments.'},
        {'name': '--x_duration', 'type': int, 'default': 1, 'help': 'Episode duration multiplier.'},
        {'name': '--end_step', 'type': int, 'default': -1, 'help': 'Max steps until auto-termination.'},
        {'name': '--ctrl_mode', 'type': int, 'default': CTRL_MAN, 'help': 'Sim/agent control mode.'},
        {'name': '--rec_mode', 'type': int, 'default': REC_NONE, 'help': 'Data category to record.'},
        {'name': '--headless', 'type': int, 'default': 0, 'help': 'Option to run without a viewer.'},
        {'name': '--act_freq', 'type': int, 'default': cfg.STEPS_PER_SECOND, 'help': 'Inference steps per second.'},
        {'name': '--model_name', 'type': str, 'default': 'mazeai', 'help': 'Model name/ID string.'},
        {'name': '--model_type', 'type': int, 'default': MDL_COM, 'help': 'Communication and guidance options.'},
        {'name': '--rng_seed', 'type': int, 'default': 42, 'help': 'Seed for numpy and torch RNGs.'}]

    def __init__(self, args: Namespace):
        self.end_step: int = args.end_step
        self.ctrl_mode: int = args.ctrl_mode
        self.rec_mode: int = args.rec_mode
        self.headless = bool(args.headless)

        self.steps_per_second: int = min(args.act_freq, 64)
        fps = self.steps_per_second if self.headless else 64
        self.inference_stride = fps // self.steps_per_second

        if not np.log2(self.steps_per_second).is_integer():
            raise ValueError(f'Action frequency is expected to be a power of 2, but {self.steps_per_second} is not.')

        # Recolouring takes a second to get through 99% of the transition
        self.rgb_retain_const = 0.01**(1./fps)
        self.rgb_update_const = 1. - self.rgb_retain_const

        # Resume model state
        device = 'cuda' if args.use_gpu_pipeline else 'cpu'
        assert device != 'cpu' and torch.cuda.is_available(), f'Unable to run CUDA graphs on {device} pipeline.'

        self.device = torch.device(device)

        self.model_options = self.MODEL_OPTIONS[args.model_type]
        model_name = f'{args.model_name}_{self.model_options["suffix"]}'

        self.ckpter = CheckpointTracker(model_name, cfg.DATA_DIR, device, args.rng_seed)
        rng = self.ckpter.rng if args.level < args.keep_level else None

        # Init IsaacGym and generate initial envs
        self.sim = maze.MazeSim(args.level, args.n_bots, args.n_envs, fps, args, rng)
        self.gym = self.sim.gym
        self.interface = Interface(self) if not self.headless else None

        # Extend or diminish standard episode duration
        self.sim.ep_duration: int = round(self.sim.ep_duration * args.x_duration)
        self.regen_envs = bool(args.regen)

        # To decorrelate experience/batches, some envs have premature first resets
        steps_in_ep = max(1, self.sim.ep_duration * self.steps_per_second)
        self.forced_rst_interval = int(np.ceil(steps_in_ep / self.sim.n_envs))
        self.forced_rst_n_envs = int(self.sim.n_envs / steps_in_ep)
        self.forced_rst_env_idx = 0
        self.forced_rst_step_ctr = 0

        # Init tracked tensors
        self.init_tensors()
        self.rst_env_indices: 'list[int]' = []
        self.rec_data_queue: 'deque[tuple[Tensor, Tensor]]' = deque()

        # Prepare computational graph components
        self.op = OpAccelerator(
            self,
            self.sim.n_envs,
            self.sim.n_bots,
            self.sim.ep_duration,
            self.steps_per_second,
            maze.GOAL_RADIUS,
            device)

        # AsyncIO is used to resume ops during long IsaacGym calls
        self.async_event_loop = asyncio.get_event_loop()
        self.async_temp_result = None

    def init_tensors(self):
        """Wrap IsaacGym components, set initial and placeholder data."""

        sim = self.sim
        gym = self.gym
        device = self.device

        # Init tensor API
        gym.prepare_sim(sim.handle)

        # Physics
        actor_states = gym.acquire_actor_root_state_tensor(sim.handle)
        dof_states = gym.acquire_dof_state_tensor(sim.handle)

        gym.refresh_dof_state_tensor(sim.handle)
        gym.refresh_actor_root_state_tensor(sim.handle)

        self.actor_states: Tensor = gymtorch.wrap_tensor(actor_states)
        self.dof_states: Tensor = gymtorch.wrap_tensor(dof_states)

        # Views of data with fixed memory address
        asr = self.actor_states.reshape(sim.n_envs, -1, 13)
        self.env_bot_pos3 = asr[:, -sim.n_bots:, :3]
        self.env_bot_pos = asr[:, -sim.n_bots:, :2]
        self.env_bot_ori = asr[:, -sim.n_bots:, 3:7]

        self.bot_vel = asr[:, -sim.n_bots:, 7:10]
        self.bot_ang_vel = asr[:, -sim.n_bots:, 10:]

        self.bot_pos = self.env_bot_pos.reshape(-1, 2)
        self.bot_ori = self.env_bot_ori.reshape(-1, 4)
        self.dof_vel = self.dof_states[:, 1].reshape(-1, 4)

        # Not this one
        self.bot_old_vel = torch.zeros((sim.n_all_bots, 3), device=device)

        # Camera
        self.img_rgb_list: 'list[Tensor]' = [
            gymtorch.wrap_tensor(gym.get_camera_image_gpu_tensor(
                sim.handle, env.handle, cam, gymapi.IMAGE_COLOR))
            for env in sim.envs
            for cam in env.cam_handles]

        self.img_dep_list: 'list[Tensor]' = [
            gymtorch.wrap_tensor(gym.get_camera_image_gpu_tensor(
                sim.handle, env.handle, cam, gymapi.IMAGE_DEPTH))
            for env in sim.envs
            for cam in env.cam_handles]

        # Segmentation images are only produced in recording modes
        self.img_seg_list: 'None | list[Tensor]' = (
            None
            if self.rec_mode == self.REC_NONE
            else [
                gymtorch.wrap_tensor(gym.get_camera_image_gpu_tensor(
                    sim.handle, env.handle, cam, gymapi.IMAGE_SEGMENTATION))
                for env in sim.envs
                for cam in env.cam_handles])

        # Objectives
        # KxLxL -> NxLxL
        self.obj_trans_prob = np.stack([env.data.obj_trans_probs for env in sim.envs])
        self.obj_trans_prob = np.repeat(self.obj_trans_prob, sim.n_bots, axis=0)
        self.obj_trans_prob = torch.from_numpy(self.obj_trans_prob).to(device, dtype=torch.float32)

        # KxCx3 (C=L) -> NxLx3
        self.obj_rgb = np.stack([maze.COLOURS['basic'][env.data.obj_clr_idcs] for env in sim.envs])
        self.obj_rgb = np.repeat(self.obj_rgb, sim.n_bots, axis=0)
        self.obj_rgb = torch.from_numpy(self.obj_rgb).to(device, dtype=torch.float32)

        # KxLx2 -> NxLx2
        self.obj_pos = np.stack([env.data.obj_points for env in sim.envs])
        self.obj_pos = np.repeat(self.obj_pos, sim.n_bots, axis=0)
        self.obj_pos = torch.from_numpy(self.obj_pos).to(device, dtype=torch.float32)

        # Sample initial tasks
        ini_bot_pos = np.concatenate([env.data.bot_spawn_points for env in sim.envs])
        ini_bot_pos = torch.from_numpy(ini_bot_pos).to(device, dtype=torch.float32)
        goal_indices = self.sample_tasks(ini_bot_pos[:, None], self.obj_pos, self.obj_trans_prob)

        # Clone not to write over reference tensors
        bot_range = np.arange(sim.n_all_bots)
        self.goal_rgb = self.obj_rgb[bot_range, goal_indices].clone()
        self.goal_pos = self.obj_pos[bot_range, goal_indices].clone()

        # Must be on cpu for custom line of sight tracer
        self.goal_pos_arr = self.goal_pos.cpu().numpy()
        self.goal_wallgrid_idx = np.digitize(self.goal_pos_arr, sim.open_grid_delims)

        self.goal_path_len = torch.zeros(sim.n_all_bots, device=device)
        self.goal_path_dir = torch.zeros((sim.n_all_bots, 2), device=device)
        self.goal_in_sight_mask = torch.zeros(sim.n_all_bots, dtype=torch.bool, device=device)

        # Task tracking
        self.env_run_times = torch.zeros(sim.n_envs, device=device)
        self.bot_time_on_task = torch.zeros(sim.n_all_bots, device=device)
        self.bot_time_at_goal = torch.zeros(sim.n_all_bots, device=device)
        self.bot_done_ctr = torch.zeros(sim.n_all_bots, device=device)
        self.throughput = torch.zeros(sim.n_all_bots, device=device)

        # Reset flags
        self.bot_done_mask = torch.zeros(sim.n_all_bots, dtype=torch.bool, device=device)
        self.bot_rst_mask = torch.zeros(sim.n_all_bots, dtype=torch.bool, device=device)

        # Action feedback
        if self.ctrl_mode == self.CTRL_GEN:
            self.actions = torch.hstack((
                torch.zeros((sim.n_all_bots, cfg.DOF_VEC_SIZE), device=device),
                torch.rand((sim.n_all_bots, cfg.RGB_VEC_SIZE), device=device)))
        else:
            self.actions = torch.zeros((sim.n_all_bots, cfg.ACT_VEC_SIZE), device=device)

        self.act_trq, self.act_rgb = torch.split(self.actions, cfg.ACT_VEC_SPLIT, dim=1)

    def reset_tensors(self) -> 'tuple[np.ndarray, np.ndarray] | None':
        """Restore tensor states after an env reset or task completion."""

        # Based on env resets
        if self.rst_env_indices:
            rst_envs = [self.sim.envs[i] for i in self.rst_env_indices]
            bot_rst_indices = torch.nonzero(self.bot_rst_mask, as_tuple=True)[0]

            obj_trans_prob = np.stack([env.data.obj_trans_probs for env in rst_envs])
            obj_trans_prob = np.repeat(obj_trans_prob, self.sim.n_bots, axis=0)
            obj_trans_prob = torch.from_numpy(obj_trans_prob).to(self.device, dtype=torch.float32)
            self.obj_trans_prob[bot_rst_indices] = obj_trans_prob

            obj_rgb = np.stack([maze.COLOURS['basic'][env.data.obj_clr_idcs] for env in rst_envs])
            obj_rgb = np.repeat(obj_rgb, self.sim.n_bots, axis=0)
            obj_rgb = torch.from_numpy(obj_rgb).to(self.device, dtype=torch.float32)
            self.obj_rgb[bot_rst_indices] = obj_rgb

            obj_pos = np.stack([env.data.obj_points for env in rst_envs])
            obj_pos = np.repeat(obj_pos, self.sim.n_bots, axis=0)
            obj_pos = torch.from_numpy(obj_pos).to(self.device, dtype=torch.float32)
            self.obj_pos[bot_rst_indices] = obj_pos

            ini_bot_pos = np.concatenate([env.data.bot_spawn_points for env in rst_envs])
            self.bot_pos[bot_rst_indices] = torch.from_numpy(ini_bot_pos).to(self.device, dtype=torch.float32)

            self.env_run_times[self.rst_env_indices] = 0.
            self.bot_done_ctr[bot_rst_indices] = 0.

            self.bot_old_vel[bot_rst_indices] = 0.

            if self.ctrl_mode == self.CTRL_GEN:
                self.act_trq[bot_rst_indices] = act_trq = \
                    torch.zeros((len(bot_rst_indices), cfg.DOF_VEC_SIZE), device=self.device)

                self.act_rgb[bot_rst_indices] = act_rgb = \
                    torch.rand((len(bot_rst_indices), cfg.RGB_VEC_SIZE), device=self.device)

                self.actions[bot_rst_indices] = torch.hstack((act_trq, act_rgb))

            else:
                self.act_trq[bot_rst_indices] = 0.
                self.act_rgb[bot_rst_indices] = 0.
                self.actions[bot_rst_indices] = 0.

            bot_rclr_mask = self.bot_done_mask | self.bot_rst_mask

        else:
            bot_rclr_mask = self.bot_done_mask

        # Based on env resets and task completions
        if torch.any(bot_rclr_mask).item():
            bot_rclr_indices = torch.nonzero(bot_rclr_mask, as_tuple=True)[0]
            bot_rclr_idx_arr = bot_rclr_indices.cpu().numpy()

            new_goal_indices = self.sample_tasks(
                self.bot_pos[bot_rclr_indices, None],
                self.obj_pos[bot_rclr_indices],
                self.obj_trans_prob[bot_rclr_indices])

            self.goal_rgb[bot_rclr_indices] = goal_rgb = self.obj_rgb[bot_rclr_indices, new_goal_indices]
            self.goal_pos[bot_rclr_indices] = goal_pos = self.obj_pos[bot_rclr_indices, new_goal_indices]

            self.goal_pos_arr[bot_rclr_idx_arr] = goal_pos_arr = goal_pos.cpu().numpy()
            self.goal_wallgrid_idx[bot_rclr_idx_arr] = np.digitize(goal_pos_arr, self.sim.open_grid_delims)

            self.bot_time_on_task[bot_rclr_indices] = 0.
            self.bot_time_at_goal[bot_rclr_indices] = 0.

            return bot_rclr_idx_arr, goal_rgb.cpu().numpy()

        return None

    def eval_reset(self):
        """Regenerate envs and update the associated root states."""

        # Force premature resets
        if self.ctrl_mode == self.CTRL_RL and self.forced_rst_env_idx < self.sim.n_envs:
            self.forced_rst_step_ctr += 1

            if self.forced_rst_step_ctr >= self.forced_rst_interval:
                rst_env_slice = slice(self.forced_rst_env_idx, self.forced_rst_env_idx + self.forced_rst_n_envs)
                self.rst_env_indices = list(range(rst_env_slice.start, rst_env_slice.stop))

                rst_env_mask = torch.zeros(self.sim.n_envs, dtype=torch.bool, device=self.device)
                rst_env_mask[rst_env_slice] = True

                bot_rst_mask = torch.repeat_interleave(rst_env_mask, self.sim.n_bots)
                self.bot_rst_mask = self.bot_rst_mask | bot_rst_mask

                self.forced_rst_env_idx += self.forced_rst_n_envs
                self.forced_rst_step_ctr = 0

        if not self.rst_env_indices:
            return

        # Get updated states
        actor_states, actor_indices = self.sim.reset(self.rst_env_indices, self.regen_envs)

        # Set new states
        gym_indices = torch.from_numpy(actor_indices).to(self.device, dtype=torch.int32)
        self.actor_states[actor_indices] = torch.from_numpy(actor_states).to(self.device, dtype=torch.float32)

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim.handle,
            gymtorch.unwrap_tensor(self.actor_states),
            gymtorch.unwrap_tensor(gym_indices),
            len(actor_indices))

        # Reset viewer perspective
        if not self.headless:
            self.interface.reset()

    def reset_all(self):
        self.bot_rst_mask = torch.ones_like(self.bot_rst_mask)
        self.rst_env_indices = list(range(self.sim.n_envs))

    def sample_tasks(
        self,
        bot_pos: Tensor,
        landmark_pos_ref: Tensor,
        landmark_probs: Tensor
    ) -> Tensor:
        """Create new assignments for agents."""

        # Uniform
        if self.ctrl_mode == self.CTRL_GEN:
            probs = torch.ones(landmark_probs.shape[:2], device=self.device)

        # Get the transition probs of the closest objective
        else:
            # Rx1x2 -> RxLx2 -> RxL
            diff = bot_pos - landmark_pos_ref
            dist = torch.linalg.norm(diff, dim=-1)

            # RxL -> R
            origin_idx = torch.argmin(dist, dim=-1)

            # RxLxL -> RxL
            probs = landmark_probs[torch.arange(len(origin_idx)), origin_idx]

        # Sample R new assignments
        task_idx = torch.multinomial(probs, 1)

        return task_idx.flatten()

    def step(
        self,
        actions: Tensor = None,
        get_info: bool = True
    ) -> 'tuple[tuple[Tensor, ...], Tensor, Tensor, dict[str, Any]]':
        """
        Apply actions in environments, evaluate their effects,
        step physics and graphics, record current data.
        """

        if actions is None:
            # Initial step
            self.set_colours(None, self.goal_rgb.cpu().numpy(), maze.BOT_CARGO_IDX)
            act_rgb = self.act_rgb

        elif self.headless:
            # Pre-physics step
            act_rgb = self.apply_actions(actions)

        else:
            # Get events
            self.interface.eval_events()
            act_rgb = self.apply_actions(actions)

            # Step physics and graphics (the last stride is made later)
            for _ in range(self.inference_stride-1):
                self.update_colours(act_rgb)
                self.set_colours()

                self.gym.simulate(self.sim.handle)
                self.gym.step_graphics(self.sim.handle)

                self.gym.fetch_results(self.sim.handle, True)
                self.interface.sync_redraw()

        self.update_colours(act_rgb)

        # Reset environments
        # NOTE: Resets are one step delayed, as all envs must be updated together
        self.eval_reset()

        # Async update tensors and colours, step physics
        self.run_async_ops(self.async_reset_and_recolour, self.async_simulate)
        rst_mask_f = self.bot_rst_mask.float()

        # Async render, compute rewards and resets from physical state
        self.run_async_ops(self.async_eval_state, self.async_step_graphics)
        obs_aux, reward = self.async_temp_result

        # Async compute observations
        # NOTE: Refreshing cameras is by far the longest part of a step
        self.run_async_ops(self.async_get_vector_observations, self.async_render_cameras)

        obs_img, obs_seg = self.get_image_observations()
        obs_vec = self.async_temp_result
        obs = (obs_img, obs_vec, obs_aux)

        # Keep data for debugging via interface
        self.async_temp_result = (obs_vec, obs_aux)

        # Keep data to save
        if self.rec_mode:
            self.update_rec_data_queue(
                obs_seg,
                *obs,
                reward[:, None],
                rst_mask_f[:, None],
                self.act_trq,
                self.act_rgb)

        # Externally handled log
        info = {'score': self.get_throughput()} if get_info else self.NULL_INFO

        return obs, reward, rst_mask_f, info

    def apply_actions(self, actions: Tensor) -> Tensor:
        """Set torques, relay RGB signals."""

        self.actions = actions
        self.act_trq, act_rgb = torch.split(actions, cfg.ACT_VEC_SPLIT, dim=1)

        self.act_trq = torch.clamp(self.act_trq, -maze.MOT_MAX_TORQUE, maze.MOT_MAX_TORQUE)
        act_rgb = torch.clamp(act_rgb, 0., 1.)

        act_trq = self.act_trq.reshape(-1)
        self.gym.set_dof_actuation_force_tensor(self.sim.handle, gymtorch.unwrap_tensor(act_trq))

        return act_rgb

    def update_colours(self, act_rgb: Tensor):
        """Update colour transition by an exponentially weighted moving average."""

        self.act_rgb = self.rgb_retain_const * self.act_rgb + self.rgb_update_const * act_rgb

    def get_throughput(self) -> float:
        """Get the average number of tasks completed per minute."""

        return self.throughput.mean().item()

    def update_rec_data_queue(self, seg: Tensor, img: Tensor, *vecs: 'tuple[Tensor]'):
        img_data = torch.cat((img, seg[:, None].float()), dim=1).cpu().numpy()
        vec_data = torch.hstack(vecs).cpu().numpy()

        # Images are stored in full or vector form (as means)
        if self.rec_mode == self.REC_VEC:
            img_data = np.mean(img_data, axis=(-2, -1))

        self.rec_data_queue.append((img_data, vec_data))

    def save_rec_data(self):
        if self.rec_mode == self.REC_NONE or not self.rec_data_queue:
            return

        file_idx = get_available_file_idx(cfg.DATA_DIR, 'rec')
        filename = os.path.join(cfg.DATA_DIR, f'rec_{file_idx:02d}.npz')

        img = np.stack([x[0] for x in self.rec_data_queue])
        vec = np.stack([x[1] for x in self.rec_data_queue])
        self.rec_data_queue.clear()

        np.savez_compressed(filename, img=img, vec=vec)

        print(f'Saved data to: {filename}')

    def get_image_observations(self) -> Tensor:
        self.gym.start_access_image_tensors(self.sim.handle)

        obs_img = self.op.get_image_observations(self.img_rgb_list, self.img_dep_list)
        obs_seg = None if self.img_seg_list is None else torch.stack(self.img_seg_list)

        self.gym.end_access_image_tensors(self.sim.handle)

        return obs_img, obs_seg

    def set_colours(
        self,
        indices: np.ndarray = None,
        colours: np.ndarray = None,
        body_idx: int = maze.BOT_BODY_IDX
    ):
        handles = self.sim.env_bot_handles if indices is None else self.sim.env_bot_handles[indices]
        colours = self.act_rgb.cpu().numpy() if colours is None else colours

        for (env_handle, agent_handle), rgb in zip(handles, colours):
            self.gym.set_rigid_body_color(
                env_handle, agent_handle, body_idx, gymapi.MESH_VISUAL, gymapi.Vec3(*rgb))

    async def async_reset_and_recolour(self):
        rclr_data = self.reset_tensors()

        # Colouring action
        self.set_colours()

        # Recolour goal indicator
        if rclr_data is None:
            return

        self.set_colours(*rclr_data, maze.BOT_CARGO_IDX)

    async def async_simulate(self, other_task: asyncio.Task):
        self.gym.simulate(self.sim.handle)

        await other_task
        self.async_event_loop.stop()

    async def async_eval_state(self):
        """Get state data, compute rewards, check terminal conditions."""

        # Update tensor data
        self.gym.fetch_results(self.sim.handle, True)
        self.gym.refresh_actor_root_state_tensor(self.sim.handle)
        self.gym.refresh_dof_state_tensor(self.sim.handle)

        self.bot_pos.copy_(self.env_bot_pos.reshape(-1, 2))
        self.bot_ori.copy_(self.env_bot_ori.reshape(-1, 4))

        bot_pos_arr = self.bot_pos.cpu().numpy()

        # Check line of sight
        goal_in_sight_mask = eval_line_of_sight(
            bot_pos_arr,
            self.goal_pos_arr,
            self.goal_wallgrid_idx,
            self.sim.open_grid_delims,
            self.sim.all_wallgrid_pairs)

        self.goal_in_sight_mask.copy_(torch.from_numpy(goal_in_sight_mask))

        # Estimate shortest path
        slices = (slice(i, i+self.sim.n_bots) for i in range(0, self.sim.n_all_bots, self.sim.n_bots))

        path_res = [
            env.data.get_path_estimate(bot_pos_arr[idcs], self.goal_pos_arr[idcs], goal_in_sight_mask[idcs])
            for env, idcs in zip(self.sim.envs, slices)]

        goal_path_len = np.concatenate([path_res_i[0] for path_res_i in path_res])
        goal_path_dir = np.concatenate([path_res_i[1] for path_res_i in path_res])

        self.goal_path_len.copy_(torch.from_numpy(goal_path_len))
        self.goal_path_dir.copy_(torch.from_numpy(goal_path_dir))

        # Eval rewards and resets
        (
            self.env_run_times,
            self.bot_time_on_task,
            self.bot_time_at_goal,
            self.throughput,
            self.bot_done_ctr,
            self.bot_done_mask,
            self.bot_rst_mask,
            rst_env_mask,
            reward,
            obs_aux
        ) = self.op.eval_state(
                self.bot_pos,
                self.bot_ori,
                self.goal_pos,
                self.goal_path_len,
                self.goal_path_dir,
                self.goal_in_sight_mask,
                self.env_run_times,
                self.bot_time_on_task,
                self.bot_time_at_goal,
                self.bot_done_ctr)

        self.rst_env_indices = torch.nonzero(rst_env_mask, as_tuple=True)[0].tolist()
        self.async_temp_result = (obs_aux, reward)

    async def async_step_graphics(self, other_task: asyncio.Task):
        self.gym.step_graphics(self.sim.handle)

        if not self.headless:
            self.interface.sync_redraw()

        await other_task
        self.async_event_loop.stop()

    async def async_get_vector_observations(self):
        self.async_temp_result = self.op.get_vector_observations(
            self.bot_pos,
            self.bot_vel,
            self.bot_old_vel,
            self.bot_ori,
            self.bot_ang_vel,
            self.dof_vel,
            self.act_trq,
            self.act_rgb,
            self.goal_rgb,
            self.bot_time_at_goal,
            self.throughput)

    async def async_render_cameras(self, other_task: asyncio.Task):
        if self.ctrl_mode > self.CTRL_MAN or self.rec_mode > self.REC_NONE:
            self.gym.render_all_camera_sensors(self.sim.handle)

        await other_task
        self.async_event_loop.stop()

    def run_async_ops(self, fn_a: Callable, fn_b: Callable):
        task_a = self.async_event_loop.create_task(fn_a())
        task_b = self.async_event_loop.create_task(fn_b(task_a))

        try:
            self.async_event_loop.run_forever()

        except KeyboardInterrupt as interrupt:
            task_a.cancel()
            task_b.cancel()

            try:
                self.async_event_loop.run_until_complete(task_a if task_a.exception() is None else task_b)

            except asyncio.CancelledError:
                pass

            raise interrupt

    def run(self):
        print(f'Data logging is {"ON" if self.rec_mode else "OFF"}.')

        try:
            if self.ctrl_mode == self.CTRL_RL:
                self.train()

            elif self.ctrl_mode == self.CTRL_AI:
                self.eval()

            else:
                self.play()

            print('Ending...')

        except KeyboardInterrupt:
            print('Interrupted...')

        self.save_rec_data()

        # Cleanup
        if self.interface is not None:
            self.gym.destroy_viewer(self.interface.viewer)

        self.sim.cleanup()
        self.async_event_loop.close()

        print('Done.')

    def train(self):
        model = ActorCritic(self.sim.n_bots, self.model_options['com'], self.model_options['guide'])
        optimiser = NAdamW(list(model.policy.parameters()) + list(model.valuator.parameters()), lr=2e-5)

        model.to(self.ckpter.device)
        model.visencoder.load_state_dict(torch.load(os.path.join(cfg.DATA_DIR, 'visnet', 'encoder_000.pt')))
        self.ckpter.load_model(model, optimiser)

        # Accelerate collector and critic
        mem = model.init_mem(self.sim.n_all_bots, detach=True)
        model.fwd_partial = self.op.accel_action(model.fwd_partial, mem)

        scheduler = SoftConstLRScheduler(
            optimiser,
            step_milestones=cfg.UPDATE_MILESTONE_MAP[self.sim.level],
            starting_step=self.ckpter.meta['update_step'])

        # Half-life of rewards is at 1/8th of an episode
        gamma = 0.5 ** (1. / ((self.sim.ep_duration / 8) * self.steps_per_second))

        rl_algo = PPG(
            self.step,
            self.ckpter,
            scheduler,
            cfg.N_EPOCHS_MAP[self.sim.level],
            cfg.LOG_EPOCH_INTERVAL,
            cfg.CKPT_EPOCH_INTERVAL,
            cfg.BRANCH_EPOCH_INTERVAL,
            cfg.N_ROLLOUT_STEPS,
            cfg.N_TRUNCATED_STEPS,
            self.sim.n_all_bots,
            cfg.N_ROLLOUTS_PER_EPOCH,
            cfg.N_AUX_ITERS_PER_EPOCH,
            gamma,
            log_dir=cfg.LOG_DIR)

        rl_algo.run()

    def eval(self):
        model = ActorCritic(self.sim.n_bots, self.model_options['com'], self.model_options['guide'])
        self.ckpter.load_model(model)

        # Accelerate actor
        mem = model.init_mem(self.sim.n_all_bots, detach=True)
        model.fwd_partial_actor = self.op.accel_action(model.fwd_partial_actor, mem)

        # Reinit. tensors modified in-place during warm-up and capture
        mem = model.init_mem(self.sim.n_all_bots)

        with torch.inference_mode():
            obs = self.step(get_info=False)[0]
            step_ctr = -1

            while (step_ctr := step_ctr + 1) != self.end_step:
                actions, mem = model.fwd_actor(obs, mem)
                obs, _, rst_mask_f, _ = self.step(actions, get_info=False)

                if rst_mask_f.any().item():
                    mem = model.reset_mem(mem, rst_mask_f)

    def play(self):
        with torch.inference_mode():
            self.step(get_info=False)
            step_ctr = -1

            while (step_ctr := step_ctr + 1) != self.end_step:
                self.step(self.actions, get_info=False)


if __name__ == '__main__':
    args = gymutil.parse_arguments(description='Run MazeBots session.', custom_parameters=Session.ARGS)
    session = Session(args)
    session.run()
