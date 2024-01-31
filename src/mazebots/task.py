"""Rules and state transitions"""

import asyncio
from typing import Any, Callable

import numpy as np
from isaacgym import gymapi, gymtorch
import torch
from torch import Tensor
from torch.nn.functional import one_hot

from discit.accel import capture_graph
from discit.func import symlog

import config as cfg
from sim import MazeSim, BOT_BODY_IDX, BOT_CARGO_IDX, MOT_MAX_TORQUE
from utils import eval_line_of_sight
from utils_torch import (
    apply_quat_rot, check_fov, clip_angle_range, get_eulz_from_quat, norm_distance, rgb_to_hsv, weighted_sum)


# Mag. units are nT / 100
# Data for 46-3-5N | 14-30-22E | 295M | 2023-03-22
# https://www.ngdc.noaa.gov/geomag/calculators/magcalc.shtml#igrfwmm
# https://www.sensorsone.com/local-gravity-calculator/
MAG_REF = [[0.220967, 0.017385, 0.428883]]
ACC_REF = [[0., 0., -0.980624]]

# RGB receiver phases
_RAD_90 = 90./180. * torch.pi
RCVR_REL_PHASES = [[0., -_RAD_90, torch.pi, _RAD_90]]

# Expecting 1 completion per 3 seconds (edge L1 case) to 1 per 60 (optimistic L7 case) or 10 to 0.5 per 30 seconds
THROUGHPUT_WINDOW = 30.
MIN_TASK_DURATION = 3.

# Task completion confirmed after 1 second at goal
MIN_TIME_AT_GOAL = 1.

# Episodes last up to 6 min, dt is 0.0042 min
TIME_SCALE = 1. / 60.

# Max. env. distances
MAX_ENV_WIDTH = cfg.LEVEL_PARAMS[7]['env_width']
MAX_ENV_HALFWIDTH = MAX_ENV_WIDTH / 2.
MAX_IMG_DEPTH = MAX_ENV_WIDTH * 2 / 3.
ENV_DIAG_SPAN = np.floor(MAX_ENV_WIDTH * 2**0.5)

# Colour palette
COLOURS = {clr_group: np.array(clrs) for clr_group, clrs in cfg.COLOURS.items()}

# Reward params.
COMPLETION_WEIGHT = 2.
EXPLORATION_WEIGHT = 0.05  # COMPLETION_WEIGHT / (cfg.LEVEL_PARAMS[7]['n_grid_segments']**2 / 10)

COLLISION_WEIGHT = 2. * EXPLORATION_WEIGHT
PROXIMITY_WEIGHT = 2. * COLLISION_WEIGHT


class BasicInterface:
    def __init__(self, gym: gymapi.Gym, sim_handle: gymapi.Sim):
        self.gym = gym
        self.sim_handle = sim_handle
        self.viewer = gym.create_viewer(sim_handle, gymapi.CameraProperties())
        self.paused = False

        if self.viewer is None:
            raise Exception('Failed to create viewer.')

    def eval_events(self):
        if self.gym.query_viewer_has_closed(self.viewer):
            raise KeyboardInterrupt

    def sync_redraw(self, after_eval: bool = True):
        self.gym.draw_viewer(self.viewer, self.sim_handle, False)
        self.gym.sync_frame_time(self.sim_handle)

    def reset(self):
        pass


class MazeTask:
    """
    Describes the flow of the task on a tensor level.
    A subset of methods are optimised by execution with CUDA graphs.
    """

    NULL_INFO = {}
    null_obs_img: Tensor

    actor_states: Tensor
    dof_vel: Tensor
    bot_pos: Tensor
    bot_ori: Tensor
    bot_vel: Tensor
    bot_ang_vel: Tensor
    bot_old_vel: Tensor
    img_rgb_list: 'list[Tensor]'
    img_dep_list: 'list[Tensor]'
    img_seg_list: 'list[Tensor]'
    obj_trans_prob: Tensor
    obj_rgb: Tensor
    obj_pos: Tensor
    obj_idx_map: Tensor
    goal_pos_idx: Tensor
    goal_idx: Tensor
    goal_rgb: Tensor
    goal_pos: Tensor
    last_path_len: Tensor
    goal_path_len: Tensor
    goal_path_dir: Tensor
    goal_in_sight: Tensor
    obj_in_sight: Tensor
    obj_found: Tensor
    env_run_times: Tensor
    bot_time_on_task: Tensor
    bot_time_at_goal: Tensor
    obj_done_ctr: Tensor
    avg_done_ctr: Tensor
    bot_done_ctr: Tensor
    bot_done_mask: Tensor
    bot_rst_mask: Tensor
    bot_pos: Tensor
    bot_vel: Tensor
    rcvr_clr_classes: Tensor
    actions: Tensor
    act_trq: Tensor
    act_rgb: Tensor
    act_clr_idx: Tensor

    cell_exploration: Tensor
    cell_state: Tensor
    cell_delims: Tensor

    bot_pos_arr: np.ndarray
    obj_pos_arr: np.ndarray
    goal_pos_arr: np.ndarray
    obj_wallgrid_idcs: np.ndarray
    goal_wallgrid_idx: np.ndarray

    env_bot_pos3: Tensor
    env_bot_pos: Tensor
    env_bot_ori: Tensor

    mag_ref: Tensor
    acc_ref: Tensor
    quat_inv: Tensor
    rcvr_rel_phases: Tensor
    zero_column: Tensor
    sky_clr: Tensor
    row_idcs: Tensor
    speaker_mask: Tensor
    speaker_mask_f: Tensor

    def __init__(
        self,
        sim: MazeSim,
        interface: BasicInterface = None,
        steps_per_second: int = cfg.STEPS_PER_SECOND,
        frames_per_second: int = 64,
        render_cameras: bool = False,
        keep_segmentation: bool = False,
        keep_rgb_over_hsv: bool = False,
        spawn_with_random_rgb: bool = False,
        uniform_task_sampling: bool = False,
        distribute_env_resets: bool = False,
        full_env_regeneration: bool = False,
        long_range_obj_signal: bool = False,
        use_team_reward: bool = True,
        num_speakers_per_env: int = None,
        device: str = 'cuda'
    ):
        self.sim = sim
        self.gym = sim.gym
        self.interface = interface
        self.headless = interface is None
        self.render_cameras = render_cameras
        self.keep_segmentation = keep_segmentation
        self.keep_rgb_over_hsv = keep_rgb_over_hsv
        self.spawn_with_random_rgb = spawn_with_random_rgb
        self.uniform_task_sampling = uniform_task_sampling
        self.distribute_env_resets = distribute_env_resets
        self.full_env_regeneration = full_env_regeneration
        self.long_range_obj_signal = long_range_obj_signal
        self.use_team_reward = use_team_reward
        self.n_speakers = sim.n_bots if num_speakers_per_env is None else min(sim.n_bots, num_speakers_per_env)

        self.rst_env_indices: 'list[int]' = []

        steps_per_second = min(steps_per_second, frames_per_second)
        frames_per_second = steps_per_second if self.headless else frames_per_second

        if frames_per_second % steps_per_second:
            raise ValueError(f'Mismatch between FPS ({frames_per_second}) and action frequency ({steps_per_second}).')

        self.inference_stride = frames_per_second // steps_per_second
        self.dt = 1. / steps_per_second
        self.steps_in_ep = max(1, sim.ep_duration * steps_per_second)

        # Init tracked tensors
        self.device = torch.device(device)
        self.init_tensors()

        # AsyncIO is used to resume ops during long IsaacGym calls
        self.async_event_loop = asyncio.get_event_loop()
        self.async_temp_result = None

        # Store for captured graph data
        self.graphs: 'dict[str, dict[str, tuple[Tensor, ...] | torch.cuda.CUDAGraph | Callable]]' = {}

    def accelerate(self):
        if self.device.type == 'cpu' or not torch.cuda.is_available():
            raise Exception(f'Unable to run CUDA graphs on {self.device} pipeline.')

        env_run_times = self.env_run_times.clone()

        # Graph 1
        self.eval_state, self.graphs['eval_state'] = capture_graph(
            self.eval_state,
            (self.act_trq, self.act_clr_idx),
            copy_idcs_out=(2,))  # Copy reward; obs. copied later in exp. collection

        # Reset tensors modified in-place during warm-up
        self.env_run_times.copy_(env_run_times)
        self.bot_time_on_task.zero_()
        self.bot_time_at_goal.zero_()
        self.bot_done_ctr.zero_()
        self.bot_done_mask.zero_()
        self.bot_rst_mask.zero_()
        self.last_path_len.zero_()
        self.obj_found.zero_()
        self.obj_done_ctr.zero_()
        self.avg_done_ctr.zero_()
        self.bot_old_vel.zero_()

        # Graph 2
        if not self.render_cameras:
            return

        self.prepare_images, self.graphs['prepare_images'] = capture_graph(
            self.prepare_images,
            (),
            copy_idcs_in=(),
            copy_idcs_out=())

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
        net_contacts = gym.acquire_net_contact_force_tensor(sim.handle)

        gym.refresh_dof_state_tensor(sim.handle)
        gym.refresh_actor_root_state_tensor(sim.handle)
        gym.refresh_net_contact_force_tensor(sim.handle)

        self.actor_states: Tensor = gymtorch.wrap_tensor(actor_states)
        self.dof_states: Tensor = gymtorch.wrap_tensor(dof_states)
        self.net_contacts: Tensor = gymtorch.wrap_tensor(net_contacts)

        self.collider_indices = torch.tensor([
            gym.find_actor_rigid_body_index(env.handle, bot_handle, 'body', gymapi.DOMAIN_SIM)
            for env in sim.envs
            for bot_handle in env.bot_handles], dtype=torch.int64, device=device)

        # Views of data with fixed memory address
        asr = self.actor_states.reshape(sim.n_envs, -1, 13)
        self.env_bot_pos3 = asr[:, -sim.n_bots:, :3]
        self.env_bot_pos = asr[:, -sim.n_bots:, :2]
        self.env_bot_ori = asr[:, -sim.n_bots:, 3:7]

        self.bot_vel = asr[:, -sim.n_bots:, 7:10]
        self.bot_ang_vel = asr[:, -sim.n_bots:, 10:]
        self.dof_vel = self.dof_states[:, 1].reshape(-1, 4)

        # Not views any more, must be copied into
        self.bot_pos = self.env_bot_pos.reshape(-1, 2).contiguous()
        self.bot_ori = self.env_bot_ori.reshape(-1, 4).contiguous()

        self.bot_old_vel = torch.zeros((sim.n_all_bots, 3), device=device)

        self.bot_pos_arr = self.bot_pos.cpu().numpy()

        # Camera
        self.null_obs_img = torch.tensor(np.nan, device=device).expand(sim.n_all_bots, cfg.OBS_IMG_CHANNELS, 1, 1)

        if self.render_cameras:
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

            self.img_seg_list: 'list[Tensor]' = [
                gymtorch.wrap_tensor(gym.get_camera_image_gpu_tensor(
                    sim.handle, env.handle, cam, gymapi.IMAGE_SEGMENTATION))
                for env in sim.envs
                for cam in env.cam_handles]

        else:
            self.img_rgb_list = self.img_dep_list = self.img_seg_list = []

        # Static tensors
        self.mag_ref = torch.tensor(MAG_REF, dtype=torch.float32, device=device)
        self.acc_ref = torch.tensor(ACC_REF, dtype=torch.float32, device=device)
        self.quat_inv = torch.tensor([[-1., -1., -1., 1.]], dtype=torch.float32, device=device)
        self.rcvr_rel_phases = torch.tensor(RCVR_REL_PHASES, dtype=torch.float32, device=device)
        self.zero_column = torch.zeros((sim.n_all_bots, 1), dtype=torch.float32, device=device)
        self.sky_clr = torch.tensor(COLOURS['sky'][0], dtype=torch.float32, device=device)
        self.row_idcs = torch.arange(self.sim.n_all_bots, device=device)

        self.speaker_mask = torch.zeros(sim.n_bots, dtype=torch.bool, device=device)
        self.speaker_mask[:self.n_speakers] = True
        self.speaker_mask = self.speaker_mask.repeat(sim.n_envs)
        self.speaker_mask_f = self.speaker_mask.unsqueeze(-1).float()

        # Landmarks/objectives
        # ExLxL -> NxLxL
        self.obj_trans_prob = np.stack([env.data.obj_trans_probs for env in sim.envs])
        self.obj_trans_prob = np.repeat(self.obj_trans_prob, sim.n_bots, axis=0)
        self.obj_trans_prob = torch.from_numpy(self.obj_trans_prob).to(device, dtype=torch.float32)

        # ExLx3 -> NxLx3
        self.obj_rgb = np.stack([COLOURS['basic'][env.data.obj_clr_idcs] for env in sim.envs])
        self.obj_rgb = np.repeat(self.obj_rgb, sim.n_bots, axis=0)
        self.obj_rgb = torch.from_numpy(self.obj_rgb).to(device, dtype=torch.float32)

        # ExLx2 -> NxLx2
        self.obj_pos = np.stack([env.data.obj_points for env in sim.envs])
        self.obj_pos = np.repeat(self.obj_pos, sim.n_bots, axis=0)
        self.obj_pos = torch.from_numpy(self.obj_pos).to(device, dtype=torch.float32)

        # ExL -> NxL
        self.obj_idx_map = np.stack([env.data.obj_clr_idcs for env in sim.envs])
        self.obj_idx_map = np.repeat(self.obj_idx_map, sim.n_bots, axis=0)
        self.obj_idx_map = torch.from_numpy(self.obj_idx_map).to(device, dtype=torch.int64)

        # Sample initial tasks
        self.goal_pos_idx = self.sample_tasks(uniform=True)

        # Clone not to write over reference tensors
        self.goal_idx = self.obj_idx_map[self.row_idcs, self.goal_pos_idx].clone()
        self.goal_rgb = self.obj_rgb[self.row_idcs, self.goal_pos_idx].clone()
        self.goal_pos = self.obj_pos[self.row_idcs, self.goal_pos_idx].clone()

        # Must be on cpu for custom line of sight tracer
        self.obj_pos_arr = np.moveaxis(self.obj_pos.cpu().numpy(), 1, 0)
        self.goal_pos_arr = self.goal_pos.cpu().numpy()
        self.obj_wallgrid_idcs = np.digitize(self.obj_pos_arr, sim.open_grid_delims)
        self.goal_wallgrid_idx = np.digitize(self.goal_pos_arr, sim.open_grid_delims)

        # Task tracking
        self.last_path_len = torch.zeros(sim.n_all_bots, device=device)
        self.goal_path_len = torch.zeros(sim.n_all_bots, device=device)
        self.goal_path_dir = torch.zeros((sim.n_all_bots, 2), device=device)
        self.goal_in_sight = torch.zeros(sim.n_all_bots, dtype=torch.bool, device=device)
        self.obj_in_sight = torch.zeros((sim.n_all_bots, sim.n_objects), dtype=torch.bool, device=device)
        self.obj_found = torch.zeros_like(self.obj_in_sight)

        self.env_run_times = torch.zeros(sim.n_envs, device=device)
        self.bot_time_on_task = torch.zeros(sim.n_all_bots, device=device)
        self.bot_time_at_goal = torch.zeros(sim.n_all_bots, device=device)
        self.bot_done_ctr = torch.zeros(sim.n_all_bots, device=device)
        self.avg_done_ctr = torch.zeros(sim.n_envs, device=device)
        self.obj_done_ctr = torch.zeros((sim.n_envs, cfg.N_OBJ_COLOURS), device=device)

        # Force premature first env. resets by starting their counters mid-episode
        # Having envs. at different stages helps to decorrelate experience in batches
        # NOTE: Until envs. reach a steady state, the learning rate should be low or 0.
        if self.distribute_env_resets:
            envs_per_step = sim.n_envs / self.steps_in_ep
            envs_done = 0

            for i in range(self.steps_in_ep):
                envs_to_reset = int((i+1) * envs_per_step) - envs_done

                if envs_to_reset:
                    self.env_run_times[envs_done:envs_done + envs_to_reset] = i * self.dt
                    envs_done += envs_to_reset

            self.env_run_times -= self.env_run_times[0].item()
            # self.env_run_times = self.env_run_times[torch.randperm(sim.n_envs)]

        # Reset flags
        self.bot_done_mask = torch.zeros(sim.n_all_bots, dtype=torch.bool, device=device)
        self.bot_rst_mask = torch.zeros(sim.n_all_bots, dtype=torch.bool, device=device)

        # Action feedback
        self.rcvr_clr_classes = torch.tensor(cfg.RCVR_CLR_CLASSES, device=device)

        if self.spawn_with_random_rgb:
            self.act_trq = torch.zeros((sim.n_all_bots, cfg.N_DOF_MOT), device=device)
            self.act_rgb = self.sample_colours()

            self.actions = torch.hstack((self.act_trq, self.act_rgb))

        else:
            self.actions = torch.zeros((sim.n_all_bots, cfg.ACT_SIZE), device=device)
            self.act_trq, self.act_rgb = torch.split(self.actions, cfg.ACT_SPLIT, dim=1)

        self.act_clr_idx = self.get_closest_colour_index(self.act_rgb)

        # Grid cell states
        self.cell_delims = torch.from_numpy(sim.open_grid_delims).to(device, dtype=torch.float32)
        n_segments = sim.constructor.n_supgrid_segments

        cell_idx, cell_idy = torch.bucketize(self.bot_pos, self.cell_delims).unbind(1)
        self.cell_exploration = torch.ones((sim.n_all_bots, n_segments, n_segments), device=device)
        self.cell_exploration[self.row_idcs, cell_idx, cell_idy] = 0.

        self.cell_state = torch.empty((sim.n_envs, cfg.STATE_SPA_CHANNELS-1, n_segments, n_segments), device=device)
        self.update_cell_state(sim.envs)

    def update_cell_state(self, envs: list, env_indices: 'list[int]' = None):
        """Update the cell state tensor for given envs."""

        n_segments = self.sim.constructor.n_supgrid_segments

        for i, env in enumerate(envs) if env_indices is None else zip(env_indices, envs):

            # NESW connections
            hor_con_mask_f = torch.from_numpy(~env.data.hor_wall_mask).to(torch.float32)
            ver_con_mask_f = torch.from_numpy(~env.data.ver_wall_mask).to(torch.float32)

            self.cell_state[i, 0].copy_(hor_con_mask_f[:n_segments])
            self.cell_state[i, 1].copy_(ver_con_mask_f[:, 1:n_segments+1])
            self.cell_state[i, 2].copy_(hor_con_mask_f[1:n_segments+1])
            self.cell_state[i, 3].copy_(ver_con_mask_f[:, :n_segments])

            # Hue & saturation
            # NOTE: Value omitted, almost all are 1 and therefore not a distinguishing feature
            cell_clr = torch.from_numpy(COLOURS['pastel'][env.data.sqr_roof_clr_idcs])
            cell_clr = rgb_to_hsv(cell_clr, stack_dim=0)

            self.cell_state[i, 4:6] = cell_clr[:2]

            # Object mask
            self.cell_state[:, 6:].zero_()

            obj_idx = self.obj_idx_map[i*self.sim.n_bots]
            cell_idx = self.obj_wallgrid_idcs[:, i*self.sim.n_bots, 0]
            cell_idy = self.obj_wallgrid_idcs[:, i*self.sim.n_bots, 1]

            self.cell_state[i, 6 + obj_idx, cell_idx, cell_idy] = 1.

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

            obj_rgb = np.stack([COLOURS['basic'][env.data.obj_clr_idcs] for env in rst_envs])
            obj_rgb = np.repeat(obj_rgb, self.sim.n_bots, axis=0)
            obj_rgb = torch.from_numpy(obj_rgb).to(self.device, dtype=torch.float32)
            self.obj_rgb[bot_rst_indices] = obj_rgb

            obj_pos = np.stack([env.data.obj_points for env in rst_envs])
            obj_pos = np.repeat(obj_pos, self.sim.n_bots, axis=0)
            obj_pos = torch.from_numpy(obj_pos).to(self.device, dtype=torch.float32)
            self.obj_pos[bot_rst_indices] = obj_pos

            self.obj_found[bot_rst_indices] = False

            bot_rst_idcs_arr = bot_rst_indices.cpu().numpy()
            self.obj_pos_arr[:, bot_rst_idcs_arr] = obj_pos_arr = np.moveaxis(obj_pos.cpu().numpy(), 1, 0)
            self.obj_wallgrid_idcs[:, bot_rst_idcs_arr] = np.digitize(obj_pos_arr, self.sim.open_grid_delims)

            obj_idx_map = np.stack([env.data.obj_clr_idcs for env in rst_envs])
            obj_idx_map = np.repeat(obj_idx_map, self.sim.n_bots, axis=0)
            obj_idx_map = torch.from_numpy(obj_idx_map).to(self.device, dtype=torch.int64)
            self.obj_idx_map[bot_rst_indices] = obj_idx_map

            self.update_cell_state(rst_envs, self.rst_env_indices)

            ini_bot_pos = np.concatenate([env.data.bot_spawn_points for env in rst_envs])
            self.bot_pos[bot_rst_indices] = torch.from_numpy(ini_bot_pos).to(self.device, dtype=torch.float32)

            self.env_run_times[self.rst_env_indices] = 0.
            self.obj_done_ctr[self.rst_env_indices] = 0.
            self.bot_done_ctr[bot_rst_indices] = 0.

            self.bot_old_vel[bot_rst_indices] = 0.

            if self.spawn_with_random_rgb:
                self.act_trq[bot_rst_indices] = act_trq = \
                    torch.zeros((len(bot_rst_indices), cfg.N_DOF_MOT), device=self.device)

                self.act_rgb[bot_rst_indices] = act_rgb = self.sample_colours(len(bot_rst_indices))
                self.act_clr_idx[bot_rst_indices] = self.get_closest_colour_index(act_rgb)

                self.actions[bot_rst_indices] = torch.hstack((act_trq, act_rgb))

            else:
                self.act_trq[bot_rst_indices] = 0.
                self.act_rgb[bot_rst_indices] = 0.
                self.act_clr_idx[bot_rst_indices] = 0
                self.actions[bot_rst_indices] = 0.

            bot_rclr_mask = self.bot_done_mask | self.bot_rst_mask

        else:
            bot_rclr_mask = self.bot_done_mask

        # Based on env resets and task completions
        if torch.any(bot_rclr_mask).item():
            bot_rclr_indices = torch.nonzero(bot_rclr_mask, as_tuple=True)[0]
            bot_rclr_idx_arr = bot_rclr_indices.cpu().numpy()

            goal_pos_idx = self.sample_tasks(bot_rclr_indices, self.uniform_task_sampling)

            self.goal_pos_idx[bot_rclr_indices] = goal_pos_idx
            self.goal_idx[bot_rclr_indices] = self.obj_idx_map[bot_rclr_indices, goal_pos_idx]
            self.goal_rgb[bot_rclr_indices] = goal_rgb = self.obj_rgb[bot_rclr_indices, goal_pos_idx]
            self.goal_pos[bot_rclr_indices] = goal_pos = self.obj_pos[bot_rclr_indices, goal_pos_idx]
            self.goal_path_len[bot_rclr_indices] = 0.

            self.goal_pos_arr[bot_rclr_idx_arr] = goal_pos_arr = goal_pos.cpu().numpy()
            self.goal_wallgrid_idx[bot_rclr_idx_arr] = np.digitize(goal_pos_arr, self.sim.open_grid_delims)

            self.bot_time_on_task[bot_rclr_indices] = 0.
            self.bot_time_at_goal[bot_rclr_indices] = 0.

            cell_idx, cell_idy = torch.bucketize(self.bot_pos[bot_rclr_indices], self.cell_delims).unbind(1)
            self.cell_exploration[bot_rclr_indices] = 1.
            self.cell_exploration[bot_rclr_indices, cell_idx, cell_idy] = 0.

            return bot_rclr_idx_arr, goal_rgb.cpu().numpy()

        return None

    def eval_reset(self):
        """Regenerate envs and update the associated root states."""

        if not self.rst_env_indices:
            return

        # Get updated states
        actor_states, actor_indices = self.sim.reset(self.rst_env_indices, self.full_env_regeneration)

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
        self.bot_rst_mask.fill_(True)
        self.rst_env_indices = list(range(self.sim.n_envs))

    def sample_colours(self, n_to_sample: int = None) -> Tensor:
        if n_to_sample is None:
            n_to_sample = self.sim.n_all_bots

        clr_indices = torch.randint(cfg.N_RCVR_CLR_CLASSES, (n_to_sample,), device=self.device)

        return self.rcvr_clr_classes.index_select(0, clr_indices)

    def sample_tasks(self, indices: Tensor = None, uniform: bool = False) -> Tensor:
        """Create new assignments for agents."""

        if uniform:
            n_to_sample = self.sim.n_all_bots if indices is None else len(indices)

            return torch.randint(self.sim.n_objects, (n_to_sample,), device=self.device)

        # Get the transition probs for the current objective
        # RxLxL -> RxL
        if indices is None:
            probs = self.obj_trans_prob[self.row_idcs, self.goal_pos_idx]

        else:
            probs = self.obj_trans_prob[indices, self.goal_pos_idx[indices]]

        # Sample R new assignments
        task_idx = torch.multinomial(probs, 1)

        return task_idx.flatten()

    def step(
        self,
        actions: Tensor = None,
        action_indices: Tensor = None,
        get_info: bool = True
    ) -> 'tuple[tuple[Tensor, ...], Tensor, Tensor, Tensor, dict[str, Any]]':
        """
        Apply actions in environments, evaluate their effects,
        step physics and graphics, gather observations and other data.
        """

        if actions is None:
            # Initial step
            self.set_colours(None, self.goal_rgb.cpu().numpy(), BOT_CARGO_IDX)

        elif self.headless:
            # Pre-physics step
            self.apply_actions(actions, action_indices)

        else:
            while self.interface.paused:
                self.interface.eval_events()
                self.interface.sync_redraw(after_eval=False)

            # Get events
            self.interface.eval_events()
            self.apply_actions(actions, action_indices)

            # Step physics and graphics (the last stride is made later)
            for _ in range(self.inference_stride-1):
                self.set_colours()

                self.gym.simulate(self.sim.handle)
                self.gym.step_graphics(self.sim.handle)

                self.gym.fetch_results(self.sim.handle, True)
                self.interface.sync_redraw(after_eval=False)

        # Reset environments
        # NOTE: Resets and final rewards are evaluated one step ahead
        # to ensure that setter fns., followed by a sim. step, take effect before getting new observations,
        # at the cost of viewing the final actions taken in flagged environments as arbitrary (which they generally are)
        self.eval_reset()

        # Async update tensors and colours, step physics
        self.run_async_ops(self.async_reset_and_recolour, self.async_simulate)
        rst_mask_f = self.bot_rst_mask.unsqueeze(-1).float()

        # Async compute observations and rewards from physical state
        # NOTE: Refreshing cameras is by far the longest part of a step
        self.run_async_ops(self.async_eval_state, self.async_step_graphics)
        obs_vec, obs_spa, reward = self.async_temp_result

        obs_img = self.get_image_observations()
        obs = (obs_img, obs_vec, obs_spa)

        # Auxiliary value targets
        vals = obs_vec[:, cfg.AUX_VAL_SLICE].clone()

        # Externally handled log
        info = {'score': self.get_score()} if get_info else self.NULL_INFO

        return obs, reward, rst_mask_f, vals, info

    def get_score(self) -> float:
        """Get the average number of tasks completed (per bot)."""

        return self.avg_done_ctr.mean().item()

    def apply_actions(self, actions: Tensor, action_indices: Tensor = None):
        """Set torques, relay RGB signals."""

        self.actions = actions
        self.act_trq, self.act_rgb = torch.split(actions, cfg.ACT_SPLIT, dim=1)

        if action_indices is None:
            action_indices = self.get_closest_colour_index(self.act_rgb)

        # NOTE: Default is short-range (white) signal, not silence (black)
        self.act_trq = torch.clamp(self.act_trq, -MOT_MAX_TORQUE, MOT_MAX_TORQUE)
        self.act_rgb = torch.where(self.speaker_mask.unsqueeze(-1), torch.clamp(self.act_rgb, 0., 1.), 1.)
        self.act_clr_idx = torch.where(self.speaker_mask, action_indices, 1)

        act_trq = self.act_trq.reshape(-1)
        self.gym.set_dof_actuation_force_tensor(self.sim.handle, gymtorch.unwrap_tensor(act_trq))

    def get_closest_colour_index(self, rgb: Tensor) -> Tensor:
        return torch.linalg.norm(rgb.unsqueeze(1) - self.rcvr_clr_classes.unsqueeze(0), dim=-1).argmin(-1)

    def set_colours(
        self,
        indices: np.ndarray = None,
        colours: np.ndarray = None,
        body_idx: int = BOT_BODY_IDX
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

        self.set_colours(*rclr_data, BOT_CARGO_IDX)

    async def async_simulate(self, other_task: asyncio.Task):
        self.gym.simulate(self.sim.handle)

        await other_task
        self.async_event_loop.stop()

    async def async_eval_state(self):
        """Get state data, compute rewards, check terminal conditions."""

        sim = self.sim

        # Update tensor data
        self.gym.fetch_results(sim.handle, True)
        self.gym.refresh_actor_root_state_tensor(sim.handle)
        self.gym.refresh_dof_state_tensor(sim.handle)
        self.gym.refresh_net_contact_force_tensor(sim.handle)

        self.bot_pos.copy_(self.env_bot_pos.reshape(-1, 2))
        self.bot_ori.copy_(self.env_bot_ori.reshape(-1, 4))

        self.bot_pos_arr = bot_pos_arr = self.bot_pos.cpu().numpy()

        # Check line of sight
        goal_in_sight = eval_line_of_sight(
            bot_pos_arr,
            self.goal_pos_arr,
            self.goal_wallgrid_idx,
            sim.open_grid_delims,
            sim.all_wallgrid_pairs)

        self.goal_in_sight.copy_(torch.from_numpy(goal_in_sight))

        obj_in_sight = np.stack([
            eval_line_of_sight(
                bot_pos_arr,
                obj_pos_arr,
                obj_wallgrid_idx,
                sim.open_grid_delims,
                sim.all_wallgrid_pairs)
            for obj_pos_arr, obj_wallgrid_idx in zip(self.obj_pos_arr, self.obj_wallgrid_idcs)], axis=-1)

        self.obj_in_sight.copy_(torch.from_numpy(obj_in_sight))

        # Estimate shortest path
        slices = [slice(i, i+sim.n_bots) for i in range(0, sim.n_all_bots, sim.n_bots)]

        path_res = [
            env.data.get_path_estimate(
                bot_pos_arr[idcs],
                self.goal_pos_arr[idcs],
                goal_in_sight[idcs],
                sim.is_preset)
            for env, idcs in zip(sim.envs, slices)]

        goal_path_len = np.concatenate([path_res_i[0] for path_res_i in path_res])
        goal_path_dir = np.concatenate([path_res_i[1] for path_res_i in path_res])

        self.last_path_len.copy_(self.goal_path_len)
        self.goal_path_len.copy_(torch.from_numpy(goal_path_len))
        self.goal_path_dir.copy_(torch.from_numpy(goal_path_dir))

        # Eval rewards and resets
        obs_vec, obs_spa, reward, self.bot_done_mask, rst_env_mask = \
            self.eval_state(self.act_trq, self.act_clr_idx)

        self.rst_env_indices = torch.nonzero(rst_env_mask, as_tuple=True)[0].tolist()

        self.async_temp_result = (obs_vec, obs_spa, reward)

    def eval_state(self, act_trq: Tensor, act_clr_idx: Tensor) -> 'tuple[Tensor, ...]':
        n_envs, n_bots, n_all_bots = self.sim.n_envs, self.sim.n_bots, self.sim.n_all_bots
        loc_ori = self.bot_ori * self.quat_inv

        # Get distance and direction to goal objective
        goal_diff = self.goal_pos - self.bot_pos
        goal_dist: Tensor = torch.linalg.norm(goal_diff, dim=1)

        goal_proximity = 1. - goal_dist / ENV_DIAG_SPAN
        goal_path_proximity = 1. - self.goal_path_len / ENV_DIAG_SPAN

        # Rotate directions to local frame
        goal_diff = torch.cat((goal_diff, self.zero_column), dim=-1)
        goal_path_dir = torch.cat((self.goal_path_dir, self.zero_column), dim=-1)

        goal_diff = apply_quat_rot(loc_ori, goal_diff)[:, :2]
        goal_path_dir = apply_quat_rot(loc_ori, goal_path_dir)[:, :2]

        goal_dir = goal_diff / torch.linalg.norm(goal_diff, dim=1, keepdim=True)
        goal_path_dir = goal_path_dir / torch.linalg.norm(goal_path_dir, dim=1, keepdim=True)

        # Check if any goals are in reach and aimed at
        goal_in_frame = self.goal_in_sight & check_fov(goal_diff)
        goal_in_reach = (goal_dist < cfg.GOAL_RADIUS) & goal_in_frame

        # Step time
        # NOTE: In-place ops avoid needless copying between graph outputs and inputs
        self.env_run_times.add_(self.dt)
        self.bot_time_on_task.add_(self.dt)
        self.bot_time_at_goal.add_(self.dt).mul_(goal_in_reach)

        # Confirm completed tasks
        bot_done_mask = self.bot_time_at_goal >= MIN_TIME_AT_GOAL
        bot_done_mask_f = bot_done_mask.float()
        self.bot_done_ctr.add_(bot_done_mask_f)

        goal_mask_f = one_hot(self.goal_idx, cfg.N_OBJ_COLOURS).float()
        goal_done_mask_f = goal_mask_f * bot_done_mask_f.unsqueeze(-1)
        self.obj_done_ctr += goal_done_mask_f.reshape(n_envs, n_bots, -1).sum(1)

        scaled_run_time = self.env_run_times.clip(MIN_TASK_DURATION) / THROUGHPUT_WINDOW
        obj_throughput = self.obj_done_ctr / scaled_run_time.unsqueeze(-1) * (self.sim.n_objects / n_bots)

        avg_done_ctr = self.bot_done_ctr.reshape(n_envs, -1).mean(-1)
        avg_throughput = avg_done_ctr / scaled_run_time

        scaled_run_time = scaled_run_time.repeat_interleave(n_bots, output_size=n_all_bots)
        throughput = self.bot_done_ctr / scaled_run_time

        # Check for terminal envs. ahead of next sim./eval. step
        time_left = self.sim.ep_duration - self.env_run_times
        rst_env_mask = time_left <= self.dt

        bot_rst_mask = rst_env_mask.repeat_interleave(n_bots, output_size=n_all_bots)

        # Get observations
        obj_in_frame, obj_diff, obj_dist, obj_prox, obj_masked_pos = self.get_obj_data(loc_ori)
        near_src_ahead, src_dist, obs_sound = self.get_sound_data(obj_diff, obj_dist, act_clr_idx)
        min_src_dist = src_dist.min(dim=-1)[0]

        bot_vel = self.bot_vel.reshape(-1, 3).contiguous()
        bot_vel_norm = torch.linalg.norm(bot_vel[:, :2], dim=-1)
        obs_imu = self.get_imu_data(bot_vel, loc_ori)

        act_clr_mask_f = one_hot(act_clr_idx, cfg.N_RCVR_CLR_CLASSES).float()
        heur_clr_idx = self.get_heur_data(obj_in_frame, obj_dist, near_src_ahead)

        cell_idx, cell_idy = torch.bucketize(self.bot_pos, self.cell_delims).unbind(1)
        new_cell_found = self.cell_exploration[self.row_idcs, cell_idx, cell_idy]
        self.cell_exploration[self.row_idcs, cell_idx, cell_idy] *= 0.

        n_segments = self.sim.constructor.n_supgrid_segments
        total_exploration = self.cell_exploration.reshape(n_envs, n_bots, n_segments, n_segments).mean(1, keepdim=True)
        obs_spa = torch.cat((self.cell_state, total_exploration), dim=1)

        goal_delta = torch.where(self.last_path_len == 0., 0., self.last_path_len - self.goal_path_len)

        # Get rewards
        if self.use_team_reward:
            main_reward = self.bot_rst_mask * self.avg_done_ctr.repeat_interleave(n_bots, output_size=n_all_bots)

        else:
            main_reward = bot_done_mask_f * COMPLETION_WEIGHT + new_cell_found * EXPLORATION_WEIGHT

        colliding = self.net_contacts[self.collider_indices, :2].abs().sum(-1) != 0.
        near_src_prox_sum = (1. - src_dist / (2.*cfg.BOT_WIDTH)).clip(0.).sum(-1)

        # If envs. have reset, suppress rewards for incorrect (unbelonging) state/obs.
        aux_reward = ~self.bot_rst_mask * (colliding * COLLISION_WEIGHT + near_src_prox_sum * PROXIMITY_WEIGHT)

        reward = torch.stack((main_reward, -aux_reward), dim=-1)

        # Assemble vec. inputs
        env_stats = torch.stack((
            min_src_dist,
            bot_vel_norm,
            goal_delta,
            self.goal_path_len / ENV_DIAG_SPAN,
            self.bot_time_on_task * TIME_SCALE
        ), dim=-1).reshape(n_envs, n_bots, -1).mean(1)

        env_stats = torch.cat((
            # Bot avg. (5+1)
            env_stats,
            avg_throughput.unsqueeze(-1),
            # Common (9+1)
            obj_throughput,
            time_left.unsqueeze(-1) * TIME_SCALE
        ), dim=-1).repeat_interleave(n_bots, dim=0, output_size=n_all_bots)

        obs_vec = torch.cat((
            # Main vec. obs. (94 = 21+12+61)
            # DOF, IMU, AHRS, & GPS (4+9+4+2+2)
            act_trq,
            *obs_imu,
            self.bot_ori,
            self.bot_pos / MAX_ENV_HALFWIDTH,
            bot_vel[:, :2],
            # Task specification (9+3)
            goal_mask_f,
            self.speaker_mask_f,
            self.bot_time_at_goal.unsqueeze(-1),
            throughput.unsqueeze(-1),
            # Communication (11+50)
            act_clr_mask_f,
            obs_sound,
            # Hidden state (68 = 37+11+4+16); mainly intended for the critic for better value estimation
            # Aux. targets, incl. obj. in frame, obj. masked coords., & obj. proximity (1+4*9)
            goal_in_frame.unsqueeze(-1).float(),
            self.zero_column.expand(-1, 4*cfg.N_OBJ_COLOURS),
            # Task state & progress (11)
            goal_dir,
            goal_proximity.unsqueeze(-1),
            goal_path_dir,
            goal_path_proximity.unsqueeze(-1),
            self.goal_pos / MAX_ENV_HALFWIDTH,
            goal_delta.unsqueeze(-1),
            bot_done_mask_f.unsqueeze(-1),
            self.bot_time_on_task.unsqueeze(-1) * TIME_SCALE,
            # Aux. reward info (4)
            new_cell_found.unsqueeze(-1),
            near_src_ahead.unsqueeze(-1).float(),
            near_src_prox_sum.unsqueeze(-1),
            colliding.unsqueeze(-1),
            # Bot avg. & common env. stats (16)
            env_stats,
            # Other (1); mainly intended for a communication heuristic
            heur_clr_idx.unsqueeze(-1).float()
        ), dim=-1)

        # Set according to original indices for all objectives
        obj_in_frame = obj_in_frame.float()

        for i in range(self.sim.n_objects):
            obj_idx = self.obj_idx_map[:, i]
            obs_vec[self.row_idcs, obj_idx + cfg.OBS_VEC_SIZE] = obj_in_frame[:, i]
            obs_vec[self.row_idcs, obj_idx + (cfg.OBS_VEC_SIZE + cfg.N_OBJ_COLOURS)] = obj_masked_pos[:, i, 0]
            obs_vec[self.row_idcs, obj_idx + (cfg.OBS_VEC_SIZE + 2*cfg.N_OBJ_COLOURS)] = obj_masked_pos[:, i, 1]
            obs_vec[self.row_idcs, obj_idx + (cfg.OBS_VEC_SIZE + 3*cfg.N_OBJ_COLOURS)] = obj_prox[:, i]

        self.bot_old_vel.copy_(bot_vel)
        self.bot_rst_mask.copy_(bot_rst_mask)
        self.avg_done_ctr.copy_(avg_done_ctr)

        return obs_vec, obs_spa, reward, bot_done_mask, rst_env_mask

    def get_obj_data(self, loc_ori: Tensor) -> 'tuple[Tensor, ...]':
        """Get mask, direction, and distance/proximity to each objective."""

        n_objects = self.sim.n_objects

        # NxLx2, Nx2 -> NxLx2 - Nx1x2 -> NxLx2
        obj_diff = self.obj_pos - self.bot_pos[:, None]

        # NxLx2 -> NxLx3
        zero_padding = self.zero_column.unsqueeze(1).expand(-1, n_objects, -1)
        obj_diff3 = torch.cat((obj_diff, zero_padding), dim=-1)

        # Nx4, NxLx3 -> (N*L)x4, (N*L)x3
        loc_ori = loc_ori.repeat_interleave(n_objects, dim=0, output_size=self.sim.n_all_bots * n_objects)
        obj_diff3 = obj_diff3.reshape(-1, 3)

        # (N*L)x4, (N*L)x3 -> (N*L)x3 -> N*L
        loc_diff3 = apply_quat_rot(loc_ori, obj_diff3)
        obj_in_fov = check_fov(loc_diff3)

        # NxL, N*L -> NxL
        obj_in_frame = self.obj_in_sight & obj_in_fov.reshape(self.sim.n_all_bots, n_objects)

        # NxLx2 -> NxL
        obj_dist = torch.linalg.norm(obj_diff, dim=-1)
        obj_prox = 1. - obj_dist / ENV_DIAG_SPAN

        # NxL, NxLx2 -> NxLx2
        obj_masked_pos = self.obj_pos / MAX_ENV_HALFWIDTH

        if not self.long_range_obj_signal:
            self.obj_found |= obj_in_frame & (obj_dist < 3*cfg.GOAL_RADIUS)
            obj_masked_pos = torch.where(self.obj_found.unsqueeze(-1), obj_masked_pos, 0.)

        return obj_in_frame, obj_diff, obj_dist, obj_prox, obj_masked_pos

    def get_sound_data(self, obj_diff: Tensor, obj_dist: Tensor, act_clr_idx: Tensor) -> 'tuple[Tensor, ...]':
        """
        Weight signal transmissions by strength (proximity) and incoming angle
        wrt. 4 oriented receivers, each covering an angle of 90 degrees.
        """

        # Pairwise distances
        # Nx2 -> ExBx2 -> Ex1xBx2 - ExBx1x2 -> ExBxBx2 -> NxBx2
        bot_pos = self.bot_pos.reshape(self.sim.n_envs, -1, 2)
        bot_diff = bot_pos.unsqueeze(1) - bot_pos.unsqueeze(2)
        bot_diff = bot_diff.reshape(self.sim.n_all_bots, -1, 2)

        # NxBx2, NxL -> NxBx1, NxLx1
        bot_dist = torch.linalg.norm(bot_diff, dim=-1, keepdim=True)
        obj_dist = obj_dist.unsqueeze(-1)

        # Suppress own signal in distance calculation
        bot_dist = torch.where(bot_dist == 0., torch.inf, bot_dist)

        # Suppress obj. signal beyond noise threshold
        if not self.long_range_obj_signal:
            obj_dist = torch.where(obj_dist > 2*cfg.GOAL_RADIUS, torch.inf, obj_dist)

        # Proximity weights via clipped inverse prop. characteristic
        bot_wsignal_strength = 0.1 / bot_dist.clip(cfg.BOT_RADIUS)
        bot_csignal_strength = 1. / bot_dist.clip(cfg.BOT_RADIUS)
        obj_csignal_strength = (1. if self.long_range_obj_signal else 0.05) / obj_dist.clip(cfg.OBJECT_RADIUS)

        # Angle-range weights
        # NxBx2, NxLx2 -> Nx(B+L)x2
        src_diff = torch.cat((bot_diff, obj_diff), dim=1)

        # NOTE: Own angles are arbitrary, but their weight should already be suppressed
        incoming_angle = torch.atan2(src_diff[..., 1], src_diff[..., 0])
        rcvr_angles = clip_angle_range(get_eulz_from_quat(self.bot_ori) + self.rcvr_rel_phases)

        # Nx(B+L), Nx4 -> Nx(B+L)x1 - Nx1x4 -> Nx(B+L)x4
        rel_angles = incoming_angle.unsqueeze(-1) - rcvr_angles.unsqueeze(1)
        rel_angles = clip_angle_range(rel_angles)

        # Phase-shifted filters
        rcvr_weights = torch.cos(rel_angles).clip(0.)

        # Nx(BxL)x4 -> NxBx4, NxLx4
        bot_rcvr_weights, obj_rcvr_weights = rcvr_weights[:, :self.sim.n_bots], rcvr_weights[:, self.sim.n_bots:]

        # Combined weights
        # NxB|Lx4 * NxB|Lx1 -> NxB|Lx4
        bot_wsignal_strengths = bot_rcvr_weights * bot_wsignal_strength
        bot_csignal_strengths = bot_rcvr_weights * bot_csignal_strength
        obj_csignal_strengths = obj_rcvr_weights * obj_csignal_strength

        # Source detection
        # NxBx4, NxBx1 -> NxB -> N
        src_dist = bot_dist.squeeze(-1)
        near_src_ahead = ((bot_rcvr_weights[..., 0] > 0.) & (src_dist < 3*cfg.BOT_WIDTH)).any(-1)

        # Indices into mask
        bot_signal_mask = one_hot(act_clr_idx, cfg.N_RCVR_CLR_CLASSES).float()
        obj_signal_mask = one_hot(self.obj_idx_map, cfg.N_OBJ_COLOURS).float()

        # Weighted sum
        # NxV, NxBx4 -> NxVx4
        bot_wsignal_sum = weighted_sum(bot_signal_mask[:, 1:2], bot_wsignal_strengths, self.sim.n_envs)
        bot_csignal_sum = weighted_sum(bot_signal_mask[:, 2:], bot_csignal_strengths, self.sim.n_envs)

        # Own feedback
        # bot_wsignal_sum += bot_signal_mask[:, 1:2, None] * (0.1 / (3*cfg.BOT_WIDTH))
        # bot_csignal_sum += bot_signal_mask[:, 2:, None] * (1. / ENV_DIAG_SPAN)

        # NxLxV, NxLx4 -> NxVx4
        obj_csignal_sum = torch.einsum('nsv,nsw->nvw', obj_signal_mask, obj_csignal_strengths)

        # Concatenate, normalise, log norm
        signal_sums = torch.cat((bot_wsignal_sum, bot_csignal_sum + obj_csignal_sum), dim=1)
        norm_of_sums = torch.linalg.norm(signal_sums, dim=-1, keepdim=True)

        signal_sums = signal_sums / norm_of_sums.clip(1e-6)
        norm_of_sums = symlog(norm_of_sums)

        # Add norm to flattened signal vector
        # NxVx4, NxVx1 -> NxVx(4+1) -> Nx(V*5)
        signal_vec = torch.cat((signal_sums, norm_of_sums), dim=-1).flatten(1)

        return near_src_ahead, src_dist, signal_vec

    def get_heur_data(self, obj_in_frame: Tensor, obj_dist: Tensor, near_src_ahead: Tensor) -> Tensor:
        """Obj. colour if seen and near, white if obstructed, black otherwise."""

        # Expel values for objs. out of frame
        obj_dist = obj_dist + ~obj_in_frame * ENV_DIAG_SPAN

        nearest_obj_dist, nearest_obj_idx = torch.min(obj_dist, dim=-1)
        nearest_obj_idx = self.obj_idx_map[self.row_idcs, nearest_obj_idx]

        heur_clr_idx = torch.where(
            nearest_obj_dist < 3*cfg.GOAL_RADIUS,
            nearest_obj_idx + 2,
            near_src_ahead.long())

        return heur_clr_idx

    def get_imu_data(self, bot_vel: Tensor, loc_ori: Tensor) -> 'tuple[Tensor, ...]':
        """
        Simulate angular velocity, acceleration, and magnetic flux density,
        as if reported by the gyroscope, accelerometer, and magnetometer
        of a 9-DoF inertial measurement unit.
        """

        acc = (bot_vel - self.bot_old_vel) / self.dt

        mag = apply_quat_rot(loc_ori, self.mag_ref)
        acc = apply_quat_rot(loc_ori, self.acc_ref + acc)

        # Expecting at most |0. - 0.3m/0.25s| / 0.25s at abrupt stop from full speed
        acc = acc / 5.

        # Expecting at most one turn per 6 seconds
        ang_vel = self.bot_ang_vel.reshape(-1, 3)

        return ang_vel, acc, mag

    async def async_step_graphics(self, other_task: asyncio.Task):
        self.gym.step_graphics(self.sim.handle)

        if not self.headless:
            self.interface.sync_redraw()

        if self.render_cameras:
            self.gym.render_all_camera_sensors(self.sim.handle)

        await other_task
        self.async_event_loop.stop()

    def get_image_observations(self) -> Tensor:
        if not self.render_cameras:
            return self.null_obs_img

        self.gym.start_access_image_tensors(self.sim.handle)

        obs_img = self.prepare_images()

        self.gym.end_access_image_tensors(self.sim.handle)

        return obs_img

    def prepare_images(self) -> Tensor:
        rgb = torch.stack(self.img_rgb_list)[..., :3]
        dep = torch.stack(self.img_dep_list)

        # Normalise
        rgb = rgb / 255.
        dep = norm_distance(-dep, MAX_IMG_DEPTH)

        # Override black sky colour
        if self.sim.is_preset:
            seg = torch.stack(self.img_seg_list)
            sky_mask = (seg == cfg.SEG_CLS_NULL).unsqueeze(-1)

            rgb = torch.where(sky_mask, self.sky_clr, rgb)

        else:
            seg = None

        # Convert to HSV space and put channels before spatial dims.
        if self.keep_rgb_over_hsv:
            clr = rgb.permute(0, 3, 1, 2)

        else:
            clr = rgb_to_hsv(rgb, stack_dim=1)

        # Stack channels
        if self.keep_segmentation:
            if seg is None:
                seg = torch.stack(self.img_seg_list)

            seg = seg.unsqueeze(1).float()

            return torch.cat((clr, dep.unsqueeze(1), seg), dim=1)

        return torch.cat((clr, dep.unsqueeze(1)), dim=1)

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
