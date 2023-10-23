"""Rules and state transitions"""

import asyncio
from typing import Any, Callable

import numpy as np
from isaacgym import gymapi, gymtorch
import torch
from torch import Tensor

from discit.accel import capture_graph

import config as cfg
from sim import MazeSim, BOT_BODY_IDX, BOT_CARGO_IDX, MOT_MAX_TORQUE
from utils import eval_line_of_sight
from utils_torch import apply_quat_rot, clip_angle_range, get_eulz_from_quat, norm_depth_range, rgb_to_hsv


# Mag. units are nT / 100
# Data for 46-3-5N (46.0513889) 14-30-22E 295M on 2023-03-22
# https://www.ngdc.noaa.gov/geomag/calculators/magcalc.shtml#igrfwmm
# https://www.sensorsone.com/local-gravity-calculator/
MAG_REF = [[0.220967, 0.017385, 0.428883]]
ACC_REF = [[0., 0., -0.980624]]

# RGB receiver phases
_RAD_90 = 90./180. * torch.pi
RCVR_REL_PHASES = [[0., -_RAD_90, torch.pi, _RAD_90]]

# Limited to 10 in the worst case, but usually it should be much lower
# Implemented with tanh(x/10)*10 to keep sensitivity in the average case
SCALE_RCVR = 10.
SCALE_ACC = 10.

# https://forums.developer.nvidia.com/t/inaccuracy-of-dof-state-readings/197373
SCALE_DOF = 1./20.

# Expecting at most half of a turn per second
SCALE_ANG_VEL = 1./torch.pi

# Expecting 1 avg. completion per 3 to 60 seconds, i.e. 3 to 0.17 per 10 seconds
SCALE_THROUGHPUT = 10.

# Episodes last from 8 to 600 seconds, i.e. 0.133 to 10 minutes, dt is 0.0042 min
SCALE_TIME = 1./60.

# Task completion confirmed after 1 second at goal
T_TASK_CONFIRM = 1.

# Max norm. ranges
MAX_ENV_WIDTH = cfg.LEVEL_PARAMS[7]['env_width']
MAX_ENV_HALFWIDTH = MAX_ENV_WIDTH / 2.
MAX_IMG_DEPTH = MAX_ENV_WIDTH * 2 / 3.
ENV_DIAG_SPAN = np.floor(MAX_ENV_WIDTH * 2**0.5)
ENV_DIAG_SLOPE = 0.8

# Colour palette
COLOURS = {clr_group: np.array(clrs) for clr_group, clrs in cfg.COLOURS.items()}


class BasicInterface:
    def __init__(self, gym: gymapi.Gym, sim_handle: gymapi.Sim):
        self.gym = gym
        self.sim_handle = sim_handle
        self.viewer = gym.create_viewer(sim_handle, gymapi.CameraProperties())

        if self.viewer is None:
            raise Exception('Failed to create viewer.')

    def eval_events(self):
        if self.gym.query_viewer_has_closed(self.viewer):
            raise KeyboardInterrupt

    def sync_redraw(self):
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
    goal_rgb: Tensor
    goal_pos: Tensor
    goal_path_len: Tensor
    goal_path_dir: Tensor
    goal_in_sight_mask: Tensor
    obj_in_sight_mask: Tensor
    env_run_times: Tensor
    bot_time_on_task: Tensor
    bot_time_at_goal: Tensor
    throughput: Tensor
    bot_done_ctr: Tensor
    bot_done_mask: Tensor
    bot_rst_mask: Tensor
    bot_pos: Tensor
    bot_vel: Tensor
    rcvr_clr_classes: Tensor
    actions: Tensor
    act_trq: Tensor
    act_rgb: Tensor

    obj_pos_arr: np.ndarray
    goal_pos_arr: np.ndarray
    goal_wallgrid_idx: np.ndarray

    env_bot_pos3: Tensor
    env_bot_pos: Tensor
    env_bot_ori: Tensor

    mag_ref: Tensor
    acc_ref: Tensor
    quat_inv: Tensor
    rcvr_rel_phases: Tensor
    zero_z: Tensor
    zero_n_clrs: Tensor
    sky_clr: Tensor
    row_idcs: Tensor

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
        reward_sharing: bool = False,
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
        self.reward_sharing = reward_sharing

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
            (self.act_trq, self.act_rgb),
            copy_idcs_out=(2,))  # Copy reward; obs. copied later in exp. collection

        # Reset tensors modified in-place during warm-up
        self.env_run_times.copy_(env_run_times)
        self.bot_time_on_task.zero_()
        self.bot_time_at_goal.zero_()
        self.bot_done_ctr.zero_()
        self.throughput.zero_()
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
        self.bot_pos = self.env_bot_pos.reshape(-1, 2)
        self.bot_ori = self.env_bot_ori.reshape(-1, 4)

        self.bot_old_vel = torch.zeros((sim.n_all_bots, 3), device=device)

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
        self.zero_z = torch.zeros((sim.n_all_bots, 1), dtype=torch.float32, device=device)
        self.zero_n_clrs = torch.zeros((sim.n_all_bots, cfg.N_OBJ_COLOURS), device=device)
        self.sky_clr = torch.tensor(COLOURS['sky'][0] * 255., dtype=torch.float32, device=device)
        self.row_idcs = torch.arange(self.sim.n_all_bots, device=device)

        # Objectives
        # KxLxL -> NxLxL
        self.obj_trans_prob = np.stack([env.data.obj_trans_probs for env in sim.envs])
        self.obj_trans_prob = np.repeat(self.obj_trans_prob, sim.n_bots, axis=0)
        self.obj_trans_prob = torch.from_numpy(self.obj_trans_prob).to(device, dtype=torch.float32)

        # KxCx3 (C=L) -> NxLx3
        self.obj_rgb = np.stack([COLOURS['basic'][env.data.obj_clr_idcs] for env in sim.envs])
        self.obj_rgb = np.repeat(self.obj_rgb, sim.n_bots, axis=0)
        self.obj_rgb = torch.from_numpy(self.obj_rgb).to(device, dtype=torch.float32)

        # KxLx2 -> NxLx2
        self.obj_pos = np.stack([env.data.obj_points for env in sim.envs])
        self.obj_pos = np.repeat(self.obj_pos, sim.n_bots, axis=0)
        self.obj_pos = torch.from_numpy(self.obj_pos).to(device, dtype=torch.float32)

        self.obj_idx_map = np.stack([env.data.obj_clr_idcs for env in sim.envs])
        self.obj_idx_map = np.repeat(self.obj_idx_map, sim.n_bots, axis=0)
        self.obj_idx_map = torch.from_numpy(self.obj_idx_map).to(device, dtype=torch.int64)

        # Sample initial tasks
        ini_bot_pos = np.concatenate([env.data.bot_spawn_points for env in sim.envs])
        ini_bot_pos = torch.from_numpy(ini_bot_pos).to(device, dtype=torch.float32)
        goal_indices = self.sample_tasks(ini_bot_pos[:, None], self.obj_pos, self.obj_trans_prob)

        # Clone not to write over reference tensors
        bot_range = np.arange(sim.n_all_bots)
        self.goal_rgb = self.obj_rgb[bot_range, goal_indices].clone()
        self.goal_pos = self.obj_pos[bot_range, goal_indices].clone()

        # Must be on cpu for custom line of sight tracer
        self.obj_pos_arr = np.moveaxis(self.obj_pos.cpu().numpy(), 1, 0)
        self.goal_pos_arr = self.goal_pos.cpu().numpy()
        self.goal_wallgrid_idx = np.digitize(self.goal_pos_arr, sim.open_grid_delims)

        self.goal_path_len = torch.zeros(sim.n_all_bots, device=device)
        self.goal_path_dir = torch.zeros((sim.n_all_bots, 2), device=device)
        self.goal_in_sight_mask = torch.zeros(sim.n_all_bots, dtype=torch.bool, device=device)
        self.obj_in_sight_mask = torch.zeros((sim.n_objects, sim.n_all_bots), dtype=torch.bool, device=device)

        # Task tracking
        self.env_run_times = torch.zeros(sim.n_envs, device=device)
        self.bot_time_on_task = torch.zeros(sim.n_all_bots, device=device)
        self.bot_time_at_goal = torch.zeros(sim.n_all_bots, device=device)
        self.bot_done_ctr = torch.zeros(sim.n_all_bots, device=device)
        self.throughput = torch.zeros(sim.n_all_bots, device=device)

        # Force premature first env. resets by starting their counters mid-episode
        # Having envs. at different stages helps to decorrelate experience in batches
        # NOTE: Until envs. reach a steady state, the learning rate should be low or 0.
        if self.distribute_env_resets:
            separation_interval = max(1, np.ceil(self.steps_in_ep / sim.n_envs)) * self.dt
            envs_per_reset = max(1, sim.n_envs // self.steps_in_ep)

            for i in range(0, sim.n_envs, envs_per_reset):
                self.env_run_times[i:i+envs_per_reset] += i * separation_interval

        # Reset flags
        self.bot_done_mask = torch.zeros(sim.n_all_bots, dtype=torch.bool, device=device)
        self.bot_rst_mask = torch.zeros(sim.n_all_bots, dtype=torch.bool, device=device)

        # Action feedback
        self.rcvr_clr_classes = torch.tensor(cfg.RCVR_CLR_CLASSES, device=device)

        if self.spawn_with_random_rgb:
            self.actions = torch.hstack((
                torch.zeros((sim.n_all_bots, cfg.DOF_VEC_SIZE), device=device),
                self.sample_colours()))
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

            obj_rgb = np.stack([COLOURS['basic'][env.data.obj_clr_idcs] for env in rst_envs])
            obj_rgb = np.repeat(obj_rgb, self.sim.n_bots, axis=0)
            obj_rgb = torch.from_numpy(obj_rgb).to(self.device, dtype=torch.float32)
            self.obj_rgb[bot_rst_indices] = obj_rgb

            obj_pos = np.stack([env.data.obj_points for env in rst_envs])
            obj_pos = np.repeat(obj_pos, self.sim.n_bots, axis=0)
            obj_pos = torch.from_numpy(obj_pos).to(self.device, dtype=torch.float32)
            self.obj_pos[bot_rst_indices] = obj_pos

            self.obj_pos_arr[:, bot_rst_indices.cpu().numpy()] = np.moveaxis(obj_pos.cpu().numpy(), 1, 0)

            obj_idx_map = np.stack([env.data.obj_clr_idcs for env in rst_envs])
            obj_idx_map = np.repeat(obj_idx_map, self.sim.n_bots, axis=0)
            obj_idx_map = torch.from_numpy(obj_idx_map).to(self.device, dtype=torch.int64)
            self.obj_idx_map[bot_rst_indices] = obj_idx_map

            ini_bot_pos = np.concatenate([env.data.bot_spawn_points for env in rst_envs])
            self.bot_pos[bot_rst_indices] = torch.from_numpy(ini_bot_pos).to(self.device, dtype=torch.float32)

            self.env_run_times[self.rst_env_indices] = 0.
            self.bot_done_ctr[bot_rst_indices] = 0.

            self.bot_old_vel[bot_rst_indices] = 0.

            if self.spawn_with_random_rgb:
                self.act_trq[bot_rst_indices] = act_trq = \
                    torch.zeros((len(bot_rst_indices), cfg.DOF_VEC_SIZE), device=self.device)

                self.act_rgb[bot_rst_indices] = act_rgb = \
                    self.sample_colours(len(bot_rst_indices))

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
        self.bot_rst_mask = torch.ones_like(self.bot_rst_mask)
        self.rst_env_indices = list(range(self.sim.n_envs))

    def sample_colours(self, n_to_sample: int = None) -> Tensor:
        if n_to_sample is None:
            n_to_sample = self.sim.n_all_bots

        clr_indices = torch.randint(cfg.N_RCVR_CLR_CLASSES, (n_to_sample,), device=self.device)

        return self.rcvr_clr_classes.index_select(0, clr_indices)

    def sample_tasks(
        self,
        bot_pos: Tensor,
        landmark_pos_ref: Tensor,
        landmark_probs: Tensor
    ) -> Tensor:
        """Create new assignments for agents."""

        # Uniform
        if self.uniform_task_sampling:
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
    ) -> 'tuple[tuple[Tensor, ...], Tensor, Tensor, tuple[Tensor, ...], dict[str, Any]]':
        """
        Apply actions in environments, evaluate their effects,
        step physics and graphics, record current data.
        """

        if actions is None:
            # Initial step
            self.set_colours(None, self.goal_rgb.cpu().numpy(), BOT_CARGO_IDX)

        elif self.headless:
            # Pre-physics step
            self.apply_actions(actions)

        else:
            # Get events
            self.interface.eval_events()
            self.apply_actions(actions)

            # Step physics and graphics (the last stride is made later)
            for _ in range(self.inference_stride-1):
                self.set_colours()

                self.gym.simulate(self.sim.handle)
                self.gym.step_graphics(self.sim.handle)

                self.gym.fetch_results(self.sim.handle, True)
                self.interface.sync_redraw()

        # Reset environments
        # NOTE: Resets are one step delayed, as all envs must be updated together
        self.eval_reset()

        # Async update tensors and colours, step physics
        self.run_async_ops(self.async_reset_and_recolour, self.async_simulate)
        rst_mask_f = self.bot_rst_mask.float()

        # Async compute observations and rewards from physical state
        # NOTE: Refreshing cameras is by far the longest part of a step
        self.run_async_ops(self.async_eval_state, self.async_step_graphics)
        obs_vec, com_weights, reward = self.async_temp_result

        obs_img = self.get_image_observations()
        obs = (obs_img, obs_vec, com_weights)

        # Auxiliary value targets
        vals = obs_vec[:, cfg.AUX_VEC_SLICE].clone()

        # Externally handled log
        info = {'score': self.get_throughput()} if get_info else self.NULL_INFO

        return obs, reward, rst_mask_f, vals, info

    def get_throughput(self) -> float:
        """Get the average number of tasks completed per minute."""

        return self.throughput.mean().item()

    def apply_actions(self, actions: Tensor):
        """Set torques, relay RGB signals."""

        self.actions = actions
        self.act_trq, self.act_rgb = torch.split(actions, cfg.ACT_VEC_SPLIT, dim=1)

        self.act_trq = torch.clamp(self.act_trq, -MOT_MAX_TORQUE, MOT_MAX_TORQUE)
        self.act_rgb = torch.clamp(self.act_rgb, 0., 1.)

        act_trq = self.act_trq.reshape(-1)
        self.gym.set_dof_actuation_force_tensor(self.sim.handle, gymtorch.unwrap_tensor(act_trq))

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

        bot_pos_arr = self.bot_pos.cpu().numpy()

        # Check line of sight
        goal_in_sight_mask = eval_line_of_sight(
            bot_pos_arr,
            self.goal_pos_arr,
            self.goal_wallgrid_idx,
            sim.open_grid_delims,
            sim.all_wallgrid_pairs)

        self.goal_in_sight_mask.copy_(torch.from_numpy(goal_in_sight_mask))

        obj_in_sight_mask = np.stack([
            eval_line_of_sight(
                bot_pos_arr,
                obj_pos_arr,
                self.goal_wallgrid_idx,
                sim.open_grid_delims,
                sim.all_wallgrid_pairs)
            for obj_pos_arr in self.obj_pos_arr])

        self.obj_in_sight_mask.copy_(torch.from_numpy(obj_in_sight_mask))

        # Estimate shortest path
        slices = (slice(i, i+sim.n_bots) for i in range(0, sim.n_all_bots, sim.n_bots))

        path_res = [
            env.data.get_path_estimate(
                bot_pos_arr[idcs],
                self.goal_pos_arr[idcs],
                goal_in_sight_mask[idcs],
                sim.is_preset)
            for env, idcs in zip(sim.envs, slices)]

        goal_path_len = np.concatenate([path_res_i[0] for path_res_i in path_res])
        goal_path_dir = np.concatenate([path_res_i[1] for path_res_i in path_res])

        self.goal_path_len.copy_(torch.from_numpy(goal_path_len))
        self.goal_path_dir.copy_(torch.from_numpy(goal_path_dir))

        # Eval rewards and resets
        obs_vec, com_weights, reward, self.bot_done_mask, self.bot_rst_mask, rst_env_mask, obj_dist = \
            self.eval_state(self.act_trq, self.act_rgb)

        self.rst_env_indices = torch.nonzero(rst_env_mask, as_tuple=True)[0].tolist()

        # Nearest objective
        near_idx = obs_vec[:, 0].long()
        near_idx = self.obj_idx_map[self.row_idcs, near_idx] + 2

        obs_vec[:, 0] = near_idx.float()

        # Auxiliary value targets
        for i in range(obj_dist.shape[1]):
            mapped_idx = self.obj_idx_map[:, i] + cfg.OBS_VEC_SIZE
            obs_vec[self.row_idcs, mapped_idx] = norm_depth_range(obj_dist[:, i], MAX_IMG_DEPTH)

        self.async_temp_result = (obs_vec, com_weights, reward)

    def eval_state(self, act_trq: Tensor, act_rgb: Tensor) -> 'tuple[Tensor, ...]':

        # Get distance and direction to each objective
        obj_diff = self.obj_pos - self.bot_pos[:, None]
        obj_dist = torch.linalg.norm(obj_diff, dim=-1)

        loc_ori = self.bot_ori * self.quat_inv

        # Add offsets wrt. line of sight and field of view
        for i, obj_in_sight_mask in enumerate(self.obj_in_sight_mask):
            diff_vec = torch.cat((obj_diff[:, i], self.zero_z), dim=-1)
            loc_diff = apply_quat_rot(loc_ori, diff_vec)

            obj_in_fov_mask = torch.atan(loc_diff[:, 1] / loc_diff[:, 0].clip(1e-6)).abs() < (torch.pi / 4.)

            obj_dist[:, i] += (~(obj_in_sight_mask & obj_in_fov_mask)).float() * 100.

        # Heuristic
        near_dist, near_idx = torch.min(obj_dist, dim=-1)
        near_in_reach_mask = (near_dist < (3. * cfg.GOAL_RADIUS)).float()
        near_idx = near_idx.float()

        # Get distance and direction to goal objective
        goal_diff = self.goal_pos - self.bot_pos
        goal_dist: Tensor = torch.linalg.norm(goal_diff, dim=1)
        goal_dir = goal_diff / goal_dist[..., None]

        # Rotate directions to local frame
        goal_dir = torch.cat((goal_dir, self.zero_z), dim=-1)
        goal_path_dir = torch.cat((self.goal_path_dir, self.zero_z), dim=-1)

        goal_dir = apply_quat_rot(loc_ori, goal_dir)
        goal_path_dir = apply_quat_rot(loc_ori, goal_path_dir)

        # Check if any goals are in reach and aimed at
        goal_path_aim = goal_path_dir[:, 0]
        goal_path_off = goal_path_dir[:, 1]

        goal_in_fov_mask = torch.atan(goal_path_off / goal_path_aim.clip(1e-6)).abs() < (torch.pi / 4.)
        goal_in_fov_mask = self.goal_in_sight_mask & goal_in_fov_mask

        goal_in_reach_mask = (goal_dist < cfg.GOAL_RADIUS) & goal_in_fov_mask

        # Step time
        # NOTE: In-place ops avoid needless copying between graph outputs and inputs
        self.env_run_times.add_(self.dt)
        self.bot_time_on_task.add_(self.dt)
        self.bot_time_at_goal.add_(self.dt).mul_(goal_in_reach_mask.float())

        # Confirm completed tasks
        bot_done_mask = self.bot_time_at_goal >= T_TASK_CONFIRM
        bot_done_mask_f = bot_done_mask.float()
        self.bot_done_ctr.add_(bot_done_mask_f)

        throughput = \
            self.bot_done_ctr.reshape(self.sim.n_envs, -1).mean(-1) / self.env_run_times.clip(1.) * SCALE_THROUGHPUT

        # Check for terminal envs
        time_left = self.sim.ep_duration - self.env_run_times
        rst_env_mask = time_left <= 0.

        # Expand env-wise data for each agent
        time_left = time_left.repeat_interleave(self.sim.n_bots, output_size=self.sim.n_all_bots)
        bot_rst_mask = rst_env_mask.repeat_interleave(self.sim.n_bots, output_size=self.sim.n_all_bots)

        throughput = throughput.repeat_interleave(self.sim.n_bots, output_size=self.sim.n_all_bots)

        # Evalute rewards
        reward = bot_done_mask_f * (1. - self.bot_time_on_task / max(self.sim.ep_duration, self.dt))

        if self.reward_sharing:
            env_reward = reward.reshape(self.sim.n_envs, -1).sum(-1)
            env_reward = env_reward.repeat_interleave(self.sim.n_bots, output_size=self.sim.n_all_bots)

            # Score of 1 for own, score of 1/n_bots for shared rewards
            reward = reward + env_reward / self.sim.n_bots

        # Pairwise distances for auxiliary rewards and com. weights
        # Nx2 -> ExBx2 -> Ex1xBx2 - ExBx1x2 -> ExBxBx2
        pos_loc = self.bot_pos.reshape(self.sim.n_envs, -1, 2)
        pos_diff_loc = pos_loc.unsqueeze(1) - pos_loc.unsqueeze(2)

        # Set high value on diagonal to suppress own signal in distance calculation
        for pos_diff_i in pos_diff_loc:
            pos_diff_i[..., 0].fill_diagonal_(ENV_DIAG_SPAN)
            pos_diff_i[..., 1].fill_diagonal_(ENV_DIAG_SPAN)

        # ExBxBx2 -> NxBx2 -> NxBx1
        pos_diff = pos_diff_loc.reshape(self.sim.n_all_bots, -1, 2)
        dist: Tensor = torch.linalg.norm(pos_diff, dim=-1, keepdim=True)

        # Proximity weights
        prox_weights_near = norm_depth_range(dist, 1.)
        prox_weights_env = norm_depth_range(dist, ENV_DIAG_SPAN, ENV_DIAG_SLOPE)

        # Angle-range weights
        # NOTE: Own angles are arbitrary, but the weight should already be suppressed via distance
        incoming_angle = torch.atan2(pos_diff[..., 1], pos_diff[..., 0])
        rcvr_angles = clip_angle_range(get_eulz_from_quat(self.bot_ori) + self.rcvr_rel_phases)

        # NxB, Nx4 -> NxBx1 - Nx1x4 -> NxBx4
        rel_angles = incoming_angle.unsqueeze(-1) - rcvr_angles.unsqueeze(1)
        rel_angles = clip_angle_range(rel_angles)

        # Phase-shifted filters
        ang_weights = torch.cos(rel_angles).clip(0.)

        # Combined weights
        # NxBx4 * NxBx1 -> NxBx4
        com_weights = ang_weights * prox_weights_env

        # Auxiliary rewards
        prox_channels = (ang_weights * prox_weights_near).sum(1)
        path_obstruction_pen = torch.tanh(prox_channels[:, 0] - prox_channels[:, 2]).clip(0.)

        goal_proximity = norm_depth_range(goal_dist, ENV_DIAG_SPAN, ENV_DIAG_SLOPE)
        goal_path_proximity = norm_depth_range(self.goal_path_len, ENV_DIAG_SPAN, ENV_DIAG_SLOPE)
        path_alignment_rew = goal_path_proximity * goal_path_aim.clip(0.)

        colliding = (self.net_contacts[self.collider_indices, :2].abs().sum(-1) != 0.).float()
        body_contact_pen = (0.1 / self.sim.level) * colliding

        # Max (1to6 - 0 - 0), min (0 - 1to6 - 4to24); main rewards expected between (2 to 12) or (4 to 24) when sharing
        # NOTE: Aux. rewards slightly diminish through levels, as main rewards get sparser as well
        aux_reward = (0.025 * (path_alignment_rew - path_obstruction_pen) - 0.1 * body_contact_pen) / self.sim.level
        reward = reward + aux_reward

        # Vector observations
        bot_vel = self.bot_vel.reshape(-1, 3).contiguous()
        obs_imu = self.imu(bot_vel)

        # NOTE: Differencing reported DOF pos. would be less noisy
        # https://forums.developer.nvidia.com/t/inaccuracy-of-dof-state-readings/197373/5
        dof_vel = self.dof_vel * SCALE_DOF

        goal_hsv = rgb_to_hsv(self.goal_rgb)
        act_hsv = rgb_to_hsv(act_rgb)

        obs_vec = torch.cat((
            # Heuristic (2)
            near_idx.unsqueeze(-1),
            near_in_reach_mask.unsqueeze(-1),
            # Main vec. obs. (24)
            self.bot_time_at_goal.unsqueeze(-1),
            act_trq,
            dof_vel,
            obs_imu,
            goal_hsv,
            act_hsv,
            # Training aides (9+15); should only be exposed to the critic or to train aux. value estimation
            # NOTE: Distances to objects in FOV are set outside of this (accelerated) function via indexing
            self.zero_n_clrs,
            goal_in_fov_mask.unsqueeze(-1).float(),
            self.goal_pos / MAX_ENV_HALFWIDTH,
            self.bot_pos / MAX_ENV_HALFWIDTH,
            goal_dir[:, :2],
            goal_proximity.unsqueeze(-1),
            goal_path_dir[:, :2],
            goal_path_proximity.unsqueeze(-1),
            prox_channels,
            # Hidden state remainder (9)
            # NOTE: Goal inputs explicitly repeated to the critic
            colliding.unsqueeze(-1),
            goal_hsv,
            self.bot_time_at_goal.unsqueeze(-1),
            (self.bot_time_on_task * SCALE_TIME).unsqueeze(-1),
            (time_left * SCALE_TIME).unsqueeze(-1),
            bot_done_mask_f.unsqueeze(-1),
            throughput.unsqueeze(-1)), dim=-1)

        self.throughput.copy_(throughput)
        self.bot_old_vel.copy_(bot_vel)

        return obs_vec, com_weights, reward, bot_done_mask, bot_rst_mask, rst_env_mask, obj_dist

    def imu(self, bot_vel: Tensor) -> Tensor:
        """
        Simulate angular velocity, acceleration, and magnetic flux density,
        as if reported by the gyroscope, accelerometer, and magnetometer
        of a 9-DoF inertial measurement unit.
        """

        acc = (bot_vel - self.bot_old_vel) / self.dt

        loc_ori = self.bot_ori * self.quat_inv
        mag = apply_quat_rot(loc_ori, self.mag_ref)
        acc = apply_quat_rot(loc_ori, self.acc_ref + acc)

        acc = torch.tanh(acc / SCALE_ACC) * SCALE_ACC
        ang_vel = self.bot_ang_vel.reshape(-1, 3) * SCALE_ANG_VEL

        return torch.cat((ang_vel, acc, mag), axis=-1)

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

        # Override black sky colour
        if self.sim.is_preset:
            seg = torch.stack(self.img_seg_list)
            seg_mask = (seg == cfg.SEG_CLS_NULL).unsqueeze(-1).float()

            rgb = torch.lerp(rgb.float(), self.sky_clr, seg_mask)

        else:
            seg = None

        # Normalise
        rgb = rgb / 255.
        dep = norm_depth_range(-dep, MAX_IMG_DEPTH)

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
