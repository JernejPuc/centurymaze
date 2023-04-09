"""Rules and state transitions"""

import asyncio
from typing import Any, Callable

import numpy as np
from isaacgym import gymapi, gymtorch
import torch
from torch import Tensor

from discit.accel import capture_graph

import config as cfg
import maze
from utils import eval_line_of_sight
from utils_torch import apply_quat_rot, get_eulz_from_quat, norm_depth_range


# Mag. units are nT / 100
# Data for 46-3-5N (46.0513889) 14-30-22E 295M on 2023-03-22
# https://www.ngdc.noaa.gov/geomag/calculators/magcalc.shtml#igrfwmm
# https://www.sensorsone.com/local-gravity-calculator/
MAG_REF = [[0.220967, 0.017385, 0.428883]]
ACC_REF = [[0., 0., -0.980624]]

# RGB receiver phases
_RAD_120 = 120./180. * torch.pi
RCVR_REL_PHASES = [[0., -_RAD_120, _RAD_120]]
_2PI = 2.*torch.pi

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

# Max depth range
MAX_DIST = 128.


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

    actor_states: Tensor
    dof_vel: Tensor
    bot_pos: Tensor
    bot_ori: Tensor
    bot_vel: Tensor
    bot_ang_vel: Tensor
    bot_old_vel: Tensor
    img_rgb_list: 'list[Tensor]'
    img_dep_list: 'list[Tensor]'
    img_seg_list: 'list[Tensor] | None'
    obj_trans_prob: Tensor
    obj_rgb: Tensor
    obj_pos: Tensor
    goal_rgb: Tensor
    goal_pos: Tensor
    goal_path_len: Tensor
    goal_path_dir: Tensor
    goal_in_sight_mask: Tensor
    env_run_times: Tensor
    bot_time_on_task: Tensor
    bot_time_at_goal: Tensor
    throughput: Tensor
    bot_done_ctr: Tensor
    bot_done_mask: Tensor
    bot_rst_mask: Tensor
    bot_pos: Tensor
    bot_vel: Tensor
    act_trq: Tensor
    act_rgb: Tensor

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

    def __init__(
        self,
        sim: maze.MazeSim,
        interface: BasicInterface = None,
        steps_per_second: int = cfg.STEPS_PER_SECOND,
        frames_per_second: int = 64,
        render_cameras: bool = False,
        render_segmentation: bool = False,
        spawn_with_random_rgb: bool = False,
        uniform_task_sampling: bool = False,
        distribute_env_resets: bool = False,
        full_env_regeneration: bool = False,
        device: str = 'cuda'
    ):
        self.sim = sim
        self.gym = sim.gym
        self.interface = interface
        self.headless = interface is None
        self.render_cameras = render_cameras
        self.render_segmentation = render_segmentation
        self.uniform_task_sampling = uniform_task_sampling

        steps_per_second = min(steps_per_second, frames_per_second)
        frames_per_second = steps_per_second if self.headless else frames_per_second

        if frames_per_second % steps_per_second:
            raise ValueError(f'Mismatch between FPS ({frames_per_second}) and action frequency ({steps_per_second}).')

        self.inference_stride = frames_per_second // steps_per_second
        self.dt = 1. / steps_per_second

        # Recolouring takes a second to get through 99% of the transition
        self.spawn_with_random_rgb = spawn_with_random_rgb
        self.rgb_retain_const = 0.01**(1./frames_per_second)
        self.rgb_update_const = 1. - self.rgb_retain_const

        # To decorrelate experience/batches, some envs have premature first resets
        self.steps_in_ep = max(1, sim.ep_duration * steps_per_second)
        self.forced_rst_interval = int(np.ceil(self.steps_in_ep / sim.n_envs))
        self.forced_rst_n_envs = int(sim.n_envs / self.steps_in_ep)
        self.forced_rst_env_idx = 0 if distribute_env_resets else sim.n_envs
        self.forced_rst_step_ctr = 0

        self.rst_env_indices: 'list[int]' = []
        self.full_env_regeneration = full_env_regeneration

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

        # Graph 1
        inputs = (
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

        self.eval_state, self.graphs['eval_state'] = \
            capture_graph(self.eval_state, inputs, copy_idcs_in=(), copy_idcs_out=(8,))  # Copy reward

        # Reset tensors modified in-place during warm-up and capture
        self.env_run_times.zero_()
        self.bot_time_on_task.zero_()
        self.bot_time_at_goal.zero_()
        self.bot_done_ctr.zero_()

        # Graph 2
        inputs = (
            self.bot_pos,
            self.bot_vel,
            self.bot_old_vel,
            self.bot_ori,
            self.bot_ang_vel,
            self.dof_vel,
            self.act_trq,    # Copy
            self.act_rgb,    # Copy
            self.goal_rgb,
            self.bot_time_at_goal,
            self.graphs['eval_state']['out'][3])  # Throughput

        self.get_vector_observations, self.graphs['get_vector_observations'] = \
            capture_graph(self.get_vector_observations, inputs, copy_idcs_in=(6, 7), copy_idcs_out=())

        # Graph 3
        inputs = (self.img_rgb_list, self.img_dep_list)

        self._get_image_observations, self.graphs['get_image_observations'] = \
            capture_graph(self._get_image_observations, inputs, copy_idcs_in=(), copy_idcs_out=())

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
            if not self.render_segmentation
            else [
                gymtorch.wrap_tensor(gym.get_camera_image_gpu_tensor(
                    sim.handle, env.handle, cam, gymapi.IMAGE_SEGMENTATION))
                for env in sim.envs
                for cam in env.cam_handles])

        # Static tensors
        self.mag_ref = torch.tensor(MAG_REF, dtype=torch.float32, device=device)
        self.acc_ref = torch.tensor(ACC_REF, dtype=torch.float32, device=device)
        self.quat_inv = torch.tensor([[-1., -1., -1., 1.]], dtype=torch.float32, device=device)
        self.rcvr_rel_phases = torch.tensor(RCVR_REL_PHASES, dtype=torch.float32, device=device)
        self.zero_z = torch.zeros((sim.n_all_bots, 1), dtype=torch.float32, device=device)

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
        if self.spawn_with_random_rgb:
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

            if self.spawn_with_random_rgb:
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
        if self.forced_rst_env_idx < self.sim.n_envs:
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

        obs_img = self.get_image_observations()
        obs_vec = self.async_temp_result
        obs = (*obs_img, obs_vec, obs_aux)

        # Externally handled log
        info = {'score': self.get_throughput()} if get_info else self.NULL_INFO

        return obs, reward, rst_mask_f, info

    def get_throughput(self) -> float:
        """Get the average number of tasks completed per minute."""

        return self.throughput.mean().item()

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
        ) = self.eval_state(
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

    def eval_state(
        self,
        bot_pos: Tensor,
        bot_ori: Tensor,
        goal_pos: Tensor,
        goal_path_len: Tensor,
        goal_path_dir: Tensor,
        goal_in_sight_mask: Tensor,
        env_run_times: Tensor,
        bot_time_on_task: Tensor,
        bot_time_at_goal: Tensor,
        bot_done_ctr: Tensor
    ):
        # Get distance and direction to objectives
        goal_diff = goal_pos - bot_pos
        goal_dist: Tensor = torch.linalg.norm(goal_diff, dim=1)
        goal_dir = goal_diff / goal_dist[..., None]

        # Check if any goals are in reach
        goal_in_reach_mask = (goal_dist < maze.GOAL_RADIUS) & goal_in_sight_mask

        # Step time
        # NOTE: In-place ops avoid needless copying between graph outputs and inputs
        env_run_times = env_run_times.add_(self.dt)
        bot_time_on_task = bot_time_on_task.add_(self.dt)
        bot_time_at_goal = bot_time_at_goal.add_(self.dt).mul_(goal_in_reach_mask.float())

        # Confirm completed tasks
        bot_done_mask = bot_time_at_goal >= T_TASK_CONFIRM
        bot_done_mask_f = bot_done_mask.float()
        bot_done_ctr = bot_done_ctr.add_(bot_done_mask_f)

        throughput = \
            bot_done_ctr.reshape(self.sim.n_envs, -1).mean(-1) / torch.clip(env_run_times, 1.) * SCALE_THROUGHPUT

        # Check for terminal envs
        time_left = self.sim.ep_duration - env_run_times
        rst_env_mask = time_left <= 0.

        # Expand env-wise data for each agent
        time_left = torch.repeat_interleave(time_left, self.sim.n_bots, output_size=self.sim.n_all_bots)
        bot_rst_mask = torch.repeat_interleave(rst_env_mask, self.sim.n_bots, output_size=self.sim.n_all_bots)

        throughput = torch.repeat_interleave(throughput, self.sim.n_bots, output_size=self.sim.n_all_bots)
        env_done_num = bot_done_mask_f.reshape(self.sim.n_envs, -1).sum(-1)
        env_done_num = torch.repeat_interleave(env_done_num, self.sim.n_bots, output_size=self.sim.n_all_bots)

        # Evalute rewards
        # Score of 1 for own, score of 1/n_bots for shared rewards
        reward = bot_done_mask_f + env_done_num / self.sim.n_bots

        # Auxiliary rewards
        # Max (1 + 1) / steps_in_ep
        goal_proximity = norm_depth_range(goal_path_len, MAX_DIST)
        goal_aiming = goal_path_dir[:, 0]

        reward = reward + (goal_proximity + goal_aiming) / self.steps_in_ep

        # Assemble hidden state (should only be exposed to critics for better value estimation)
        goal_dir = torch.cat((goal_dir, self.zero_z), dim=-1)
        goal_path_dir = torch.cat((goal_path_dir, self.zero_z), dim=-1)

        loc_ori = bot_ori * self.quat_inv
        goal_dir = apply_quat_rot(loc_ori, goal_dir)
        goal_path_dir = apply_quat_rot(loc_ori, goal_path_dir)

        obs_aux = torch.stack([
            goal_dir[:, 0],
            goal_dir[:, 1],
            goal_aiming,
            goal_path_dir[:, 1],
            norm_depth_range(goal_dist, MAX_DIST),
            goal_proximity,
            goal_in_sight_mask.float(),
            bot_time_at_goal,
            bot_time_on_task * SCALE_TIME,
            time_left * SCALE_TIME,
            bot_done_mask_f,
            throughput], dim=1)

        return (
            env_run_times, bot_time_on_task, bot_time_at_goal,
            throughput, bot_done_ctr, bot_done_mask,
            bot_rst_mask, rst_env_mask, reward, obs_aux)

    async def async_step_graphics(self, other_task: asyncio.Task):
        self.gym.step_graphics(self.sim.handle)

        if not self.headless:
            self.interface.sync_redraw()

        await other_task
        self.async_event_loop.stop()

    async def async_get_vector_observations(self):
        self.async_temp_result = self.get_vector_observations(
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

    def get_vector_observations(
        self,
        bot_pos: Tensor,
        bot_vel: Tensor,
        bot_old_vel: Tensor,
        bot_ori: Tensor,
        bot_ang_vel: Tensor,
        dof_vel: Tensor,
        act_trq: Tensor,
        act_rgb: Tensor,
        goal_rgb: Tensor,
        bot_time_at_goal: Tensor,
        throughput: Tensor
    ) -> Tensor:

        # NOTE: Differencing reported DOF pos. would be less noisy
        # https://forums.developer.nvidia.com/t/inaccuracy-of-dof-state-readings/197373/5
        dof_vel = dof_vel * SCALE_DOF

        bot_ang_vel = bot_ang_vel.reshape(-1, 3)
        bot_vel = bot_vel.reshape(-1, 3).contiguous()

        # IMU
        acc = (bot_vel - bot_old_vel) / self.dt
        obs_imu = self.imu(bot_ang_vel, acc, bot_ori)

        # Env local rgb/pos/ori
        rgb_loc = act_rgb.reshape(self.sim.n_envs, -1, 3)
        pos_loc = bot_pos.reshape(self.sim.n_envs, -1, 2)

        rcvr_ang = get_eulz_from_quat(bot_ori) + self.rcvr_rel_phases
        rcvr_ang_loc = rcvr_ang.reshape(self.sim.n_envs, -1, 3)

        obs_rgb = torch.cat([self.synaesthesia(*rgb_pos_ori) for rgb_pos_ori in zip(rgb_loc, pos_loc, rcvr_ang_loc)])
        obs_rgb = torch.tanh(obs_rgb / SCALE_RCVR) * SCALE_RCVR

        # Assemble
        obs_vec = torch.cat((
            bot_time_at_goal[:, None], throughput[:, None],
            act_trq, dof_vel, obs_imu, act_rgb, goal_rgb, obs_rgb), dim=-1)

        bot_old_vel.copy_(bot_vel)

        return obs_vec

    def imu(self, ang_vel: Tensor, acc: Tensor, ori: Tensor) -> Tensor:
        """
        Simulate angular velocity, acceleration, and magnetic flux density,
        as if reported by the gyroscope, accelerometer, and magnetometer
        of a 9-DoF inertial measurement unit.
        """

        loc_ori = ori * self.quat_inv
        mag = apply_quat_rot(loc_ori, self.mag_ref)
        acc = apply_quat_rot(loc_ori, self.acc_ref + acc)

        acc = torch.tanh(acc / SCALE_ACC) * SCALE_ACC
        ang_vel = ang_vel * SCALE_ANG_VEL

        return torch.cat((ang_vel, acc, mag), axis=-1)

    @staticmethod
    def synaesthesia(rgb_sig: Tensor, pos: Tensor, rcvr_angles: Tensor) -> Tensor:
        """
        Weigh RGB signal transmissions of other agents by distance and sech filters
        wrt. 3 oriented receivers, each covering an angle of 120 degrees,
        to produce 3 RGB-like accumulations.

        While not quite realistic for robotic agents, this approach can be
        viewed as inferring information from 3 sound signals or pheromones
        in superposition.

        The RGB signals are also grounded by colouring the main bodies of agents
        with their emitted colour and their cargo with the colour of their target
        objective (hence union of senses).
        """

        # Pairwise differences
        pdiff_x = pos[:, :1] - pos[None, :, 0]
        pdiff_y = pos[:, 1:] - pos[None, :, 1]
        pdiff = torch.stack((pdiff_x, pdiff_y))

        # Weigh by distance, ignoring own signal
        pdist: Tensor = torch.linalg.norm(pdiff, dim=0)
        pdist.fill_diagonal_(MAX_DIST)
        pdist = norm_depth_range(pdist, MAX_DIST)

        # NxNx3
        rgb_sig = torch.einsum('ij,ik->kij', rgb_sig, pdist)

        # Weigh by phase-shifted filters
        # Own angles are arbitrary, but pdist already masked the signals in mult
        incoming_angle = torch.atan2(pdiff_y, pdiff_x).T

        # NxNx1 - 1xNx3 -> NxNx3
        rel_angles = incoming_angle[..., None] - rcvr_angles[:, None]
        rel_angles = rel_angles + ((rel_angles < -torch.pi).float() - (rel_angles > torch.pi).float()) * _2PI

        # Cosh is naturally low at 120 degrees
        rel_weights = 1. / torch.cosh(rel_angles)

        # NxNx3, NxNx3 -> Nx3x3 -> Nx9
        rgb_agg = torch.einsum('ijk,ijl->ikl', rel_weights, rgb_sig).flatten(1)

        return rgb_agg

    async def async_render_cameras(self, other_task: asyncio.Task):
        if self.render_cameras:
            self.gym.render_all_camera_sensors(self.sim.handle)

        await other_task
        self.async_event_loop.stop()

    def get_image_observations(self) -> 'tuple[Tensor, ...]':
        self.gym.start_access_image_tensors(self.sim.handle)

        obs_img = self._get_image_observations(self.img_rgb_list, self.img_dep_list)
        obs_seg = None if self.img_seg_list is None else torch.stack(self.img_seg_list)

        self.gym.end_access_image_tensors(self.sim.handle)

        return (obs_img,) if obs_seg is None else (obs_seg, obs_img)

    @staticmethod
    def _get_image_observations(images_rgb: 'list[Tensor]', images_depth: 'list[Tensor]') -> Tensor:
        rgb = torch.stack(images_rgb)[..., :3]
        dep = torch.stack(images_depth)

        # Normalise and stack channels
        rgb = rgb / 255.
        dep = norm_depth_range(-dep, MAX_DIST)

        obs_img = torch.cat((rgb, dep[..., None]), dim=-1).permute(0, 3, 1, 2)

        return obs_img

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
