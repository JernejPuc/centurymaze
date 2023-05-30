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

# Max depth range (34 * sqrt(2) / 2)
MAX_DIST = 24.


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
        signal_object_rgb: bool = False,
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
        self.render_segmentation = render_segmentation
        self.signal_object_rgb = signal_object_rgb
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

        # Recolouring takes a second to get through 99% of the transition
        self.spawn_with_random_rgb = spawn_with_random_rgb
        self.rgb_retain_const = 0.01**(1./frames_per_second)
        self.rgb_update_const = 1. - self.rgb_retain_const

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
        self.eval_state, self.graphs['eval_state'] = \
            capture_graph(self.eval_state, (), copy_idcs_in=(), copy_idcs_out=(1,))  # Copy reward

        # Reset tensors modified in-place during warm-up
        self.env_run_times.copy_(env_run_times)
        self.bot_time_on_task.zero_()
        self.bot_time_at_goal.zero_()
        self.bot_done_ctr.zero_()
        self.throughput.zero_()

        # Graph 2
        self.get_vector_observations, self.graphs['get_vector_observations'] = \
            capture_graph(self.get_vector_observations, (self.act_trq, self.act_rgb), copy_idcs_out=())

        # Graph 3
        if not self.render_cameras:
            return

        self._get_image_observations, self.graphs['get_image_observations'] = \
            capture_graph(self._get_image_observations, (), copy_idcs_in=(), copy_idcs_out=())

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
        self.null_obs_img = torch.tensor(np.nan, device=device)

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

        else:
            self.img_rgb_list = self.img_dep_list = []

        if self.render_segmentation:
            self.img_seg_list: 'list[Tensor]' = [
                gymtorch.wrap_tensor(gym.get_camera_image_gpu_tensor(
                    sim.handle, env.handle, cam, gymapi.IMAGE_SEGMENTATION))
                for env in sim.envs
                for cam in env.cam_handles]

        else:
            self.img_seg_list = []

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
        obs_vec = self.async_temp_result

        obs_img = self.get_image_observations()
        obs = (obs_img, obs_vec, obs_aux)

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
        self.gym.refresh_net_contact_force_tensor(self.sim.handle)

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
        obs_aux, reward, self.bot_done_mask, self.bot_rst_mask, rst_env_mask = self.eval_state()

        self.rst_env_indices = torch.nonzero(rst_env_mask, as_tuple=True)[0].tolist()
        self.async_temp_result = (obs_aux, reward)

    def eval_state(self) -> 'tuple[Tensor, ...]':

        # Get distance and direction to objectives
        goal_diff = self.goal_pos - self.bot_pos
        goal_dist: Tensor = torch.linalg.norm(goal_diff, dim=1)
        goal_dir = goal_diff / goal_dist[..., None]

        # Rotate directions to local frame
        goal_dir = torch.cat((goal_dir, self.zero_z), dim=-1)
        goal_path_dir = torch.cat((self.goal_path_dir, self.zero_z), dim=-1)

        loc_ori = self.bot_ori * self.quat_inv
        goal_dir = apply_quat_rot(loc_ori, goal_dir)
        goal_path_dir = apply_quat_rot(loc_ori, goal_path_dir)

        # Check if any goals are in reach and aimed at
        goal_path_aim = goal_path_dir[:, 0]
        goal_in_reach_mask = (goal_dist < maze.GOAL_RADIUS) & self.goal_in_sight_mask & (goal_path_aim > 0.)

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
        if self.reward_sharing:
            env_done_num = bot_done_mask_f.reshape(self.sim.n_envs, -1).sum(-1)
            env_done_num = env_done_num.repeat_interleave(self.sim.n_bots, output_size=self.sim.n_all_bots)

            # Score of 1 for own, score of 1/n_bots for shared rewards
            reward = bot_done_mask_f + env_done_num / self.sim.n_bots

        else:
            reward = bot_done_mask_f

        # Auxiliary rewards
        # TODO: Computation is repeated in sense_bot_colours
        bot_pos_loc = self.bot_pos.reshape(self.sim.n_envs, -1, 2)
        bot_pos_diff_loc = bot_pos_loc.unsqueeze(1) - bot_pos_loc.unsqueeze(2)

        for pos_diff_i in bot_pos_diff_loc:
            pos_diff_i[..., 0].fill_diagonal_(MAX_DIST)

        bot_pos_diff = bot_pos_diff_loc.reshape(self.sim.n_all_bots, -1, 2)
        bot_dist: Tensor = torch.linalg.norm(bot_pos_diff, dim=-1)
        bot_dist, bot_idx = bot_dist.min(-1)
        bot_pos_diff = bot_pos_diff[torch.arange(self.sim.n_all_bots, device=self.device), bot_idx]

        bot_dir = bot_pos_diff / bot_dist.unsqueeze(-1)
        bot_dir = torch.cat((bot_dir, self.zero_z), dim=-1)
        bot_dir = apply_quat_rot(loc_ori, bot_dir)

        bot_proximity = norm_depth_range(bot_dist, self.sim.env_width * 2**0.5)
        goal_path_proximity = norm_depth_range(self.goal_path_len, self.sim.env_width * 2**0.5)
        colliding = (self.net_contacts[self.collider_indices, :2].abs().sum(-1) != 0.).float()

        # Max (1 - 0 - 0), min (-1 - 1 - 2) / steps_in_ep
        aux_reward = goal_path_proximity * goal_path_aim - bot_proximity * bot_dir[:, 0].clip(0.) - colliding * 2.
        reward = reward + aux_reward / self.steps_in_ep

        # Block weight updates to features that become relevant past level 1
        if self.sim.level == 1:
            goal_dir = goal_dir * 0.
            goal_dist = goal_dist + MAX_DIST
            goal_in_sight = torch.zeros_like(goal_dist)

        else:
            goal_in_sight = self.goal_in_sight_mask.float()

        # Assemble hidden state (should only be exposed to critics for better value estimation)
        obs_aux = torch.stack([
            goal_dir[:, 0],
            goal_dir[:, 1],
            goal_path_aim,
            goal_path_dir[:, 1],
            norm_depth_range(goal_dist, MAX_DIST),
            goal_path_proximity,
            goal_in_sight,
            colliding,
            bot_proximity,
            bot_dir[:, 0],
            bot_dir[:, 1],
            self.bot_time_at_goal,
            self.bot_time_on_task * SCALE_TIME,
            time_left * SCALE_TIME,
            bot_done_mask_f,
            throughput], dim=1)

        self.throughput.copy_(throughput)

        return obs_aux, reward, bot_done_mask, bot_rst_mask, rst_env_mask

    async def async_step_graphics(self, other_task: asyncio.Task):
        self.gym.step_graphics(self.sim.handle)

        if not self.headless:
            self.interface.sync_redraw()

        await other_task
        self.async_event_loop.stop()

    async def async_get_vector_observations(self):
        self.async_temp_result = self.get_vector_observations(self.act_trq, self.act_rgb)

    def get_vector_observations(self, act_trq: Tensor, act_rgb: Tensor) -> Tensor:

        # NOTE: Differencing reported DOF pos. would be less noisy
        # https://forums.developer.nvidia.com/t/inaccuracy-of-dof-state-readings/197373/5
        dof_vel = self.dof_vel * SCALE_DOF

        # IMU
        bot_vel = self.bot_vel.reshape(-1, 3).contiguous()
        obs_imu = self.imu(bot_vel)

        # RGB sense
        rcvr_angles = get_eulz_from_quat(self.bot_ori) + self.rcvr_rel_phases
        obs_rgb = self.sense_bot_colours(act_rgb, rcvr_angles)

        # Compensate lack of vision
        if self.signal_object_rgb:
            obs_rgb = obs_rgb + self.sense_obj_colours(rcvr_angles)

        obs_rgb = torch.tanh(obs_rgb / SCALE_RCVR) * SCALE_RCVR

        # Assemble
        obs_vec = torch.cat((
            self.bot_time_at_goal[:, None], self.throughput[:, None],
            act_trq, dof_vel, obs_imu, act_rgb, self.goal_rgb, obs_rgb), dim=-1)

        self.bot_old_vel.copy_(bot_vel)

        return obs_vec

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

    def sense_bot_colours(self, act_rgb: Tensor, rcvr_angles: Tensor) -> Tensor:

        # Pairwise differences
        # Nx2 -> ExBx2 -> Ex1xBx2 - ExBx1x2 -> ExBxBx2
        pos_loc = self.bot_pos.reshape(self.sim.n_envs, -1, 2)
        pos_diff_loc = pos_loc.unsqueeze(1) - pos_loc.unsqueeze(2)

        # Set high value on diagonal to suppress own signal in distance calculation
        for pos_diff_i in pos_diff_loc:
            pos_diff_i[..., 0].fill_diagonal_(MAX_DIST)

        # ExBxBx2 -> NxBx2
        pos_diff = pos_diff_loc.reshape(self.sim.n_all_bots, -1, 2)

        # Nx3 -> Ex1xBx3 -> ExBxBx3 -> NxBx3 (repeat_interleave does not work directly)
        act_rgb = act_rgb.reshape(self.sim.n_envs, 1, -1, 3).expand(-1, self.sim.n_bots, -1, -1)
        act_rgb = act_rgb.reshape(self.sim.n_all_bots, -1, 3).contiguous()

        return self.sense_colours(act_rgb, pos_diff, rcvr_angles)

    def sense_obj_colours(self, rcvr_angles: Tensor) -> Tensor:

        # Bot-obj differences
        # NxLx2, Nx2 -> NxLx2 - Nx1x2 -> NxLx2
        pos_diff = self.obj_pos - self.bot_pos.unsqueeze(1)

        return self.sense_colours(self.obj_rgb, pos_diff, rcvr_angles)

    @staticmethod
    def sense_colours(rgb_sig: Tensor, pos_diff: Tensor, rcvr_angles: Tensor) -> Tensor:
        """
        Weigh RGB signal transmissions of other entities by distance and sech filters
        wrt. 3 oriented receivers, each covering an angle of 120 degrees,
        to produce 3 RGB-like accumulations.

        While not quite realistic for robotic agents, this approach can be
        viewed as inferring information from 3 sound signals or pheromones
        in superposition.

        NOTE: RGB signals are also grounded by colouring the main bodies of agents
        with their emitted colour and their cargo with the colour of their target
        objective.
        """

        # Weigh by distance
        # NxMx2 -> NxM
        dist: Tensor = torch.linalg.norm(pos_diff, dim=-1)
        dist = norm_depth_range(dist, MAX_DIST)

        # NxM, (1|N)xMx3 -> NxMx1 * (1|N)xMx3 -> NxMx3
        rgb_sig = dist.unsqueeze(-1) * rgb_sig

        # Weigh by phase-shifted filters
        # NOTE: Own angles are arbitrary, but the weight should already be suppressed via distance
        incoming_angle = torch.atan2(pos_diff[..., 1], pos_diff[..., 0])

        # NxM, Nx3 -> NxMx1 - Nx1x3 -> NxMx3
        rel_angles = incoming_angle.unsqueeze(-1) - rcvr_angles.unsqueeze(1)
        rel_angles = rel_angles + ((rel_angles < -torch.pi).float() - (rel_angles > torch.pi).float()) * _2PI

        # 1/cosh is naturally low at 120 degrees
        rel_weights = 1. / torch.cosh(rel_angles)

        # NxMx3, NxMx3 -> Nx3x3 -> Nx9
        rgb_agg = torch.einsum('ijk,ijl->ikl', rel_weights, rgb_sig).flatten(1)

        return rgb_agg

    async def async_render_cameras(self, other_task: asyncio.Task):
        if self.render_cameras:
            self.gym.render_all_camera_sensors(self.sim.handle)

        await other_task
        self.async_event_loop.stop()

    def get_image_observations(self) -> Tensor:
        if not self.render_cameras:
            return self.null_obs_img

        self.gym.start_access_image_tensors(self.sim.handle)

        obs_img = self._get_image_observations()

        self.gym.end_access_image_tensors(self.sim.handle)

        return obs_img

    def _get_image_observations(self) -> Tensor:
        rgb = torch.stack(self.img_rgb_list)[..., :3]
        dep = torch.stack(self.img_dep_list)

        # Normalise and stack channels
        rgb = rgb / 255.
        dep = norm_depth_range(-dep, MAX_DIST)

        if self.render_segmentation:
            tup_img = (rgb, dep[..., None], torch.stack(self.img_seg_list)[..., None].float())

        else:
            tup_img = (rgb, dep[..., None])

        return torch.cat(tup_img, dim=-1).permute(0, 3, 1, 2)

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
