"""Simulation control flow and runtime"""

import os
from argparse import Namespace
from collections import deque
from typing import Any, Callable

import numpy as np
from PIL import Image
from isaacgym import gymapi, gymutil
import torch
from torch import Tensor

from discit.accel import capture_graph
from discit.optim import NAdamW, AdaptivePlateauScheduler
from discit.rl import PPG
from discit.track import CheckpointTracker

import config as cfg
import maze
from task import BasicInterface, MazeTask, MAX_DIST, SCALE_TIME
from model import ActorCritic
from utils import get_arg_defaults, get_available_file_idx
from utils_torch import norm_depth_range, apply_quat_rot, get_eulz_from_quat


class Interface(BasicInterface):
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

    def __init__(self, session: 'Session', sim: maze.MazeSim, device: str):
        super().__init__(sim.gym, sim.handle)

        self.session = session
        self.sim = sim

        # Init viewer
        self.viewer_top_pos = gymapi.Vec3(0., sim.env_width * 0.75, sim.env_width * 0.75)
        self.viewer_top_target = gymapi.Vec3(0., 0., 0.)

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
            key: torch.tensor(val, dtype=torch.float32, device=device)
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

        self.gym.refresh_actor_root_state_tensor(self.sim_handle)

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
        self.gym.render_all_camera_sensors(self.sim_handle)
        self.gym.start_access_image_tensors(self.sim_handle)

        rgb = self.session.img_rgb_list[self.all_bot_idx][..., :3].cpu().numpy()
        dep = 255. * norm_depth_range(-self.session.img_dep_list[self.all_bot_idx], MAX_DIST).cpu().numpy()

        self.gym.end_access_image_tensors(self.sim_handle)

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
        self.gym.draw_viewer(self.viewer, self.sim_handle, False)
        self.gym.sync_frame_time(self.sim_handle)

    def reset(self):
        self.key_vec.fill(0)
        self.update_bot_view()


class Session(MazeTask):
    """
    The main process descriptor connecting environment, operational,
    and training constructs, and providing several interfacing options.
    """

    CTRL_AI = 3
    CTRL_RL = 2
    CTRL_GEN = 1
    CTRL_MAN = 0

    REC_ALL = 2
    REC_VEC = 1
    REC_NONE = 0

    MDL_GUIDE = 3
    MDL_BLIND = 2
    MDL_COM = 1
    MDL_BASE = 0

    MODEL_OPTIONS = {
        MDL_BASE: {
            'suffix': 'base',
            'vis': True,
            'com': False,
            'guide': False},
        MDL_COM: {
            'suffix': 'com',
            'vis': True,
            'com': True,
            'guide': False},
        MDL_BLIND: {
            'suffix': 'blind',
            'vis': False,
            'com': True,
            'guide': True},
        MDL_GUIDE: {
            'suffix': 'guide',
            'vis': True,
            'com': True,
            'guide': True}}

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
        {'name': '--transfer_name', 'type': str, 'default': '', 'help': 'Base model name/ID string.'},
        {'name': '--model_name', 'type': str, 'default': 'mazeai', 'help': 'Model name/ID string.'},
        {'name': '--model_type', 'type': int, 'default': MDL_COM, 'help': 'Communication and guidance options.'},
        {'name': '--rng_seed', 'type': int, 'default': 42, 'help': 'Seed for numpy and torch RNGs.'}]

    def __init__(self, args: Namespace):
        self.end_step: int = args.end_step
        self.ctrl_mode: int = args.ctrl_mode
        self.rec_mode: int = args.rec_mode
        self.rec_data_queue: 'deque[tuple[Tensor, Tensor]]' = deque()

        # Resume model state
        device = 'cuda' if args.use_gpu_pipeline else 'cpu'

        self.model_options = self.MODEL_OPTIONS[args.model_type]
        model_name = f'{args.model_name}_{self.model_options["suffix"]}'

        self.ckpter = CheckpointTracker(model_name, cfg.DATA_DIR, device, args.rng_seed, args.transfer_name, 0, True)
        rng = self.ckpter.rng if args.level < args.keep_level else None

        # Init IsaacGym and generate initial envs
        self.steps_per_second: int = min(args.act_freq, 64)
        frames_per_second = self.steps_per_second if args.headless else 64

        sim = maze.MazeSim(args.level, args.n_bots, args.n_envs, frames_per_second, args, rng)
        interface = Interface(self, sim, device) if not args.headless else None

        # Extend or diminish standard episode duration
        sim.ep_duration: int = round(sim.ep_duration * args.x_duration)

        # Prepare computational graph components
        super().__init__(
            sim,
            interface,
            self.steps_per_second,
            frames_per_second,
            render_cameras=(
                (self.ctrl_mode > self.CTRL_MAN and self.model_options['vis'])
                or self.rec_mode > self.REC_NONE),
            render_segmentation=self.rec_mode > self.REC_NONE,
            signal_object_rgb=self.model_options['guide'],
            spawn_with_random_rgb=self.ctrl_mode == self.CTRL_GEN,
            uniform_task_sampling=self.ctrl_mode == self.CTRL_GEN,
            distribute_env_resets=self.ctrl_mode == self.CTRL_RL,
            full_env_regeneration=bool(args.regen),
            reward_sharing=args.model_type == self.MDL_COM,
            device=device)

        self.accelerate()

    def post_step(
        self,
        obs: 'tuple[Tensor, ...]',
        reward: Tensor,
        rst_mask_f: Tensor,
        _info: 'dict[str, Any]'
    ) -> 'tuple[Tensor, ...]':

        # Keep data for debugging via interface
        self.async_temp_result = obs[-2:]

        # Keep data to save
        if self.rec_mode:
            self.update_rec_data_queue(
                *obs,
                reward[:, None],
                rst_mask_f[:, None])

            # Remove segmentation channel
            if self.model_options['vis']:
                return (obs[0][:, :-1], *obs[1:])

        return obs

    def update_rec_data_queue(self, img: Tensor, *vecs: 'tuple[Tensor, ...]'):

        # Images are stored in full or vector form (as means)
        if self.rec_mode == self.REC_VEC:
            img = img.mean((-2, -1))

        img_data = img.cpu().numpy()
        vec_data = torch.hstack(vecs).cpu().numpy()

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
        # Assert that encoder exists
        encoder_path = os.path.join(cfg.DATA_DIR, 'visnet', 'encoder_000.pt')

        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f'Visual encoder is missing from presumed path: {encoder_path}')

        # Load model
        model = ActorCritic(
            self.sim.n_bots,
            self.model_options['vis'],
            self.model_options['com'],
            self.model_options['guide'],
            ignore_com=not self.reward_sharing)

        optimiser = NAdamW(
            list(model.policy.parameters()) + list(model.valuator.parameters()),
            lr=get_arg_defaults(AdaptivePlateauScheduler.__init__)['lr_milestones'][0],
            weight_decay=cfg.WEIGHT_DECAY_MAP[self.sim.level])

        model.to(self.ckpter.device)
        model.visencoder.load_state_dict(torch.load(encoder_path))
        self.ckpter.load_model(model, optimiser)

        scheduler = AdaptivePlateauScheduler(
            optimiser,
            step_milestones=cfg.UPDATE_MILESTONE_MAP[self.sim.level],
            starting_step=self.ckpter.meta['update_step'])

        # Accelerate collector, recollector, and critic
        mem = model.init_mem(self.sim.n_all_bots)
        model.collect_static = self.accel_action(model.collect_static, mem)

        mem = model.init_mem(self.sim.n_all_bots)
        model.collect_copied = self.accel_action(model.collect_copied, mem, encode=False)

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
            entropy_weight=4e-3,
            log_dir=cfg.LOG_DIR)

        try:
            rl_algo.run()

        except KeyboardInterrupt:
            rl_algo.writer.close()
            raise

    def eval(self):
        if not self.headless:
            self.interface.update_top_view()

        model = ActorCritic(
            self.sim.n_bots,
            self.model_options['vis'],
            self.model_options['com'],
            self.model_options['guide'],
            prob_actor=True,
            ignore_com=not self.reward_sharing)

        model.to(self.ckpter.device)
        self.ckpter.load_model(model)

        # Accelerate actor
        mem = model.init_mem(self.sim.n_all_bots)
        model.act_partial = self.accel_action(model.act_partial, mem)

        with torch.inference_mode():
            obs = self.step(get_info=False)[0]
            step_ctr = -1

            while (step_ctr := step_ctr + 1) != self.end_step:
                actions, mem = model.act(obs, mem)

                obs, reward, rst_mask_f, _ = self.step(actions, get_info=False)
                obs = self.post_step(obs, reward, rst_mask_f, None)

                if rst_mask_f.any().item():
                    mem = model.reset_mem(mem, rst_mask_f)

    # TODO: Body repeated from eval
    def eval_(self):
        if not self.headless:
            self.interface.update_top_view()

        model = ActorCritic(
            self.sim.n_bots,
            self.model_options['vis'],
            self.model_options['com'],
            self.model_options['guide'],
            prob_actor=True,
            ignore_com=not self.reward_sharing)

        model.to(self.ckpter.device)
        self.ckpter.load_model(model)

        # Accelerate actor
        mem = model.init_mem(self.sim.n_all_bots)
        model.act_partial = self.accel_action(model.act_partial, mem)

        # Track performance, collisions, and pathing
        throughput = 0.
        tput_ctr = 0.

        contacts = 0.
        cont_ctr = 0

        # Track pathing
        file_idx = get_available_file_idx(cfg.DATA_DIR, 'pos')
        filename = os.path.join(cfg.DATA_DIR, f'pos_{file_idx:02d}.npy')
        positions = torch.zeros((self.end_step, self.sim.n_all_bots, 2), device=self.device)

        with torch.inference_mode():
            obs = self.step(get_info=False)[0]
            step_ctr = -1

            while (step_ctr := step_ctr + 1) != self.end_step:
                actions, mem = model.act(obs, mem)

                obs, reward, rst_mask_f, _ = self.step(actions, get_info=False)
                obs = self.post_step(obs, reward, rst_mask_f, None)

                if rst_mask_f.any().item():
                    mem = model.reset_mem(mem, rst_mask_f)

                tput_mask = (self.throughput != 0.).float()
                tput_ctr = tput_ctr + tput_mask
                throughput = throughput + self.throughput * tput_mask

                cont_ctr += 1
                contacts = contacts + (self.net_contacts[self.collider_indices, :2].abs().sum(-1) != 0.).float()

                positions[step_ctr] = self.bot_pos

        tput_ctr = tput_ctr.clip(1.)
        throughput = (throughput / tput_ctr).mean().item()

        cont_ctr = max(cont_ctr, 1)
        contacts = (contacts / cont_ctr).mean().item()

        positions = positions.cpu().numpy()
        np.save(filename, positions)

        print(f'\nFinal avg. throughput: {throughput:.4f}')
        print(f'\nFinal avg. contacts: {contacts:.4f}')
        print(f'\nPositions saved to {filename}.')

    def play(self):
        if not self.headless:
            self.interface.update_top_view()

        with torch.inference_mode():
            self.step(get_info=False)
            step_ctr = -1

            while (step_ctr := step_ctr + 1) != self.end_step:
                self.post_step(*self.step(self.actions, get_info=False))

    # TODO: Body repeated from play
    def play_(self):
        if not self.headless:
            self.interface.update_top_view()

        throughput = 0.
        tput_ctr = 0.

        with torch.inference_mode():
            self.step(get_info=False)
            step_ctr = -1

            while (step_ctr := step_ctr + 1) != self.end_step:
                self.post_step(*self.step(self.actions, get_info=False))

                tput_mask = (self.throughput != 0.).float()
                tput_ctr = tput_ctr + tput_mask
                throughput = throughput + self.throughput * tput_mask

        tput_ctr = tput_ctr.clip(1.)
        throughput = (throughput / tput_ctr).mean().item()

        print(f'\nFinal avg. throughput: {throughput:.4f}')

    def accel_action(
        self,
        act_fn: Callable,
        aux_tensors: 'tuple[Tensor, ...]' = None,
        encode: bool = True
    ) -> Callable:

        if aux_tensors is None:
            aux_tensors = ()

        # Graph 4
        if encode:
            inputs = (
                self.graphs['get_image_observations']['out'] if self.render_cameras else self.null_obs_img,
                self.graphs['get_vector_observations']['out'],
                self.graphs['eval_state']['out'][0],
                *[aux.detach().clone() for aux in aux_tensors])

            act_fn, self.graphs['act_partial'] = \
                capture_graph(act_fn, inputs, copy_idcs_in=tuple(range(len(inputs)-len(aux_tensors), len(inputs))))

        # Graph 5
        else:
            inputs = (
                torch.rand_like(self.graphs['act_partial']['out'][2]),
                torch.rand_like(self.graphs['act_partial']['out'][3]),
                *[aux.detach().clone() for aux in aux_tensors])

            act_fn, self.graphs['act_partial_encoded'] = capture_graph(act_fn, inputs)

        return act_fn


if __name__ == '__main__':
    args = gymutil.parse_arguments(description='Run MazeBots session.', custom_parameters=Session.ARGS)
    session = Session(args)
    session.run()
