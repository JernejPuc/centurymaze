"""Rules and state transitions"""

import asyncio

import numpy as np
from isaacgym import gymapi, gymtorch
import torch
from torch import Tensor
from torch.nn.functional import one_hot

import config as cfg
from sim import MazeEnv, MazeSim
from utils import eval_line_of_sight
from utils_torch import (
    apply_quat_rot, check_fov, clip_angle_range, get_eulz_from_quat, get_trigonz_from_quat, norm_distance, rgb_to_hsv)


# To arrays for advanced indexing
COLOURS = {clr_group: np.array(clrs) for clr_group, clrs in cfg.COLOURS.items()}


# ------------------------------------------------------------------------------
# MARK: BasicInterface

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

    def cleanup(self):
        pass


# ------------------------------------------------------------------------------
# MARK: MazeTask

class MazeTask:
    """Describes the flow of the task on a tensor level."""

    NULL_INFO = {}

    def __init__(
        self,
        sim: MazeSim,
        interface: BasicInterface = None,
        ep_duration: float = None,
        steps_per_second: int = cfg.STEPS_PER_SECOND,
        frames_per_second: int = 64,
        render_cameras: bool = False,
        keep_segmentation: bool = False,
        stagger_env_resets: bool = False,
        reward_belief_gain: bool = False,
        reward_belief_util: bool = False,
        oracular_input: bool = False,
        track_performance: bool = False,
        device: str = 'cuda'
    ):
        self.sim = sim
        self.gym = sim.gym
        self.interface = interface
        self.ep_duration = ep_duration
        self.headless = interface is None
        self.render_cameras = render_cameras
        self.keep_segmentation = keep_segmentation
        self.stagger_env_resets = stagger_env_resets
        self.belief_gain = reward_belief_gain
        self.belief_util = reward_belief_util
        self.oracular = oracular_input
        self.device = torch.device(device)

        self.steps_per_second = steps_per_second = min(steps_per_second, frames_per_second)
        frames_per_second = self.steps_per_second if self.headless else frames_per_second

        if frames_per_second % steps_per_second:
            raise ValueError(f'Mismatch between FPS ({frames_per_second}) and action frequency ({steps_per_second}).')

        self.inference_stride = frames_per_second // steps_per_second
        self.dt = 1. / steps_per_second

        # AsyncIO is used to resume ops during long Isaac Gym calls
        self.async_event_loop = asyncio.get_event_loop()
        self.async_temp_result = None

        # ----------------------------------------------------------------------
        # MARK: init_tensor_api

        gym = self.gym
        gym.prepare_sim(sim.handle)

        # Wrap Isaac Gym components, set initial and placeholder data
        actor_states = gym.acquire_actor_root_state_tensor(sim.handle)
        net_contacts = gym.acquire_net_contact_force_tensor(sim.handle)

        gym.refresh_actor_root_state_tensor(sim.handle)
        gym.refresh_net_contact_force_tensor(sim.handle)

        self.actor_states: Tensor = gymtorch.wrap_tensor(actor_states)
        self.net_contacts: Tensor = gymtorch.wrap_tensor(net_contacts)

        self.collider_idcs = torch.tensor([
            gym.find_actor_rigid_body_index(env.handle, bot_handle, 'body', gymapi.DOMAIN_SIM)
            for env in sim.envs
            for bot_handle in env.bot_handles], dtype=torch.int64, device=device)

        # Non-contiguous data must be continuously copied to variable addresses
        self.all_n_bots = torch.from_numpy(sim.all_n_bots).to(device, dtype=torch.int64)
        self.all_bot_idcs = torch.from_numpy(sim.all_bot_idcs).to(device, dtype=torch.int64)
        self.bot_states = self.actor_states[self.all_bot_idcs]

        self.bot_pos = self.bot_states[:, :2]
        self.bot_vel = self.bot_states[:, 7:9]

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

        # ----------------------------------------------------------------------
        # MARK: init_tensors

        # Static tensors
        self.quat_inv = torch.tensor([[-1., -1., -1., 1.]], dtype=torch.float32, device=device)
        self.rcvr_rel_phases = torch.tensor([[0., 0.5, 1., -0.5]], dtype=torch.float32, device=device) * torch.pi
        self.zero_column = torch.zeros((sim.n_all_bots, 1), dtype=torch.float32, device=device)
        self.sky_clr = torch.tensor(COLOURS['neutral'][cfg.SKY_CLR_IDX], dtype=torch.float32, device=device)
        self.row_idcs = torch.arange(sim.n_all_bots, device=device)
        self.row_idx_arr = np.arange(sim.n_all_bots)

        self.depth_bins = norm_distance(
            torch.arange(0., cfg.MAX_IMG_DEPTH + cfg.DEPTH_BIN_STEP, cfg.DEPTH_BIN_STEP),
            cfg.MAX_IMG_DEPTH).to(device, dtype=torch.float32)
        self.depth_bins = (self.depth_bins[:-1], self.depth_bins[1:])

        # Goals/objects (ExGx2 -> NxGx2)
        self.obj_pos = np.concatenate([self.get_padded_obj_pts(env) for env in sim.envs])
        self.obj_pos = torch.from_numpy(self.obj_pos).to(device, dtype=torch.float32)

        # Initial tasks
        self.goal_idx_arr = np.concatenate([env.data.bot_goal_map for env in sim.envs])
        self.goal_idx = torch.from_numpy(self.goal_idx_arr).to(device, dtype=torch.int64)
        self.goal_pos = self.obj_pos[self.row_idcs, self.goal_idx]
        self.goal_mask_f = one_hot(self.goal_idx, cfg.N_GOAL_CLRS).float()

        # Must be on CPU for custom line of sight tracing and path finding
        self.bot_pos_arr = self.bot_pos.cpu().numpy()
        self.obj_pos_arr = np.moveaxis(self.obj_pos.cpu().numpy(), 1, 0)
        self.goal_pos_arr = self.goal_pos.cpu().numpy()
        self.obj_cell_idcs = np.concatenate([
            np.digitize(self.obj_pos_arr[:, env.bot_slice], env.sampler.open_delims) for env in sim.envs], axis=1)

        # Task tracking
        self.cell_rwd_sum = torch.zeros(sim.n_all_bots, device=device)
        self.goal_pred_ok = torch.zeros(sim.n_all_bots, device=device)
        self.goal_path_delta = torch.zeros(sim.n_all_bots, device=device)
        self.goal_path_len = torch.zeros(sim.n_all_bots, device=device)
        self.obj_in_frame = torch.zeros((sim.n_all_bots, cfg.N_GOAL_CLRS), dtype=torch.bool, device=device)
        self.obj_found = torch.zeros((sim.n_envs, cfg.N_GOAL_CLRS), dtype=torch.bool, device=device)

        self.bot_done_mask = torch.zeros(sim.n_all_bots, dtype=torch.bool, device=device)
        self.bot_done_mask_f = torch.zeros(sim.n_all_bots, device=device)
        self.bot_rst_mask = torch.zeros(sim.n_all_bots, dtype=torch.bool, device=device)

        self.env_rst_idcs: 'list[int]' = []
        self.ep_durations = torch.tensor([env.data.ep_duration for env in sim.envs], dtype=torch.float32, device=device)

        # Global override
        if self.ep_duration is not None:
            self.ep_durations.fill_(self.ep_duration)

        self.env_max_steps = (self.ep_durations * self.steps_per_second).long()

        # Halve the remaining duration for envs. with global spawns
        global_spawn_idcs = [i for i, env in enumerate(sim.envs) if env.data.global_spawn_flag]

        if global_spawn_idcs:
            self.env_max_steps[global_spawn_idcs] //= 2

        # Force premature first env. resets by starting their counters mid-episode
        # Having envs. at different stages helps to decorrelate experience in batches
        # NOTE: Until envs. reach a steady state, the learning rate should be low or zero
        if self.stagger_env_resets:
            self.env_step_ctrs = (self.env_max_steps * torch.rand_like(self.env_max_steps, dtype=torch.float32)).long()

        else:
            self.env_step_ctrs = torch.zeros(sim.n_envs, dtype=torch.int64, device=device)

        self.scores = [0.] * sim.n_envs

        # Restricted to 1 concurrent level only
        # NOTE: Issues from level asymmetry and envs. resetting at different times
        if track_performance:
            if len(sim.envs) > 1:
                raise NotImplementedError

            sim.n_bots = sim.envs[0].sampler.n_bots
            self.time_table = torch.full((1, sim.n_envs, sim.n_bots, 2), -1, dtype=torch.int64, device=device)

        else:
            self.time_table = None

        # Action feedback
        self.actions = torch.zeros((sim.n_all_bots, sum(cfg.ACT_SPLIT)), device=device)
        self.act_trq, self.act_com = self.actions.split(cfg.ACT_SPLIT, dim=-1)

        self.speaker_mask_f = torch.from_numpy(
            np.concatenate([env.data.speaker_mask for env in sim.envs])).to(device, dtype=torch.float32)

        # Belief
        self.obj_in_mind = torch.zeros((sim.n_all_bots, cfg.N_GOAL_CLRS), dtype=torch.bool, device=device)
        self.goal_pos_in_mind = self.bot_pos

        # Grid cell states
        self.max_delims = np.linspace(-cfg.MAX_SIDE_HALFLENGTH, cfg.MAX_SIDE_HALFLENGTH, cfg.MAX_SIDE_DIVS+1)[1:-1]
        self.max_delims = torch.from_numpy(self.max_delims).to(device, dtype=torch.float32)

        self.bot_cell_idcs = torch.bucketize(self.bot_pos, self.max_delims)

        # Bot presence & exploration
        n_max_divs = len(self.max_delims) + 1

        self.cell_bot_map = torch.zeros((sim.n_all_bots, n_max_divs, n_max_divs), device=device)
        self.cell_bot_map[(self.row_idcs, *self.bot_cell_idcs.unbind(1))] = 1.
        self.cell_found_map = self.cell_bot_map.clone()

        # NWSE walls
        cell_wall_grid = np.stack([self.get_padded_wall_grid(env, n_max_divs) for env in sim.envs])
        cell_wall_mask = np.any(cell_wall_grid[..., 0, :] != cell_wall_grid[..., 1, :], axis=-1)

        self.cell_wall_map = np.stack((
            cell_wall_mask[:, :n_max_divs, :n_max_divs, cfg.SIDE_N_IDX],
            cell_wall_mask[:, :n_max_divs, :n_max_divs, cfg.SIDE_W_IDX],
            cell_wall_mask[:, 1:n_max_divs+1, :n_max_divs, cfg.SIDE_N_IDX],
            cell_wall_mask[:, :n_max_divs, 1:n_max_divs+1, cfg.SIDE_W_IDX]), axis=1)

        self.cell_wall_map = torch.from_numpy(self.cell_wall_map).to(torch.float32)

        # Clr. cls. segmentation
        cell_cidx_map = np.stack([self.get_padded_clr_grid(env, n_max_divs) for env in sim.envs])
        cell_cidx_map = torch.from_numpy(cell_cidx_map).to(device, dtype=torch.float32)
        self.cell_seg_map = torch.where(cell_cidx_map >= 0., (cell_cidx_map + 1.) / cfg.N_WALL_CLRS, -1.)

        # Obj. presence
        self.cell_layout = torch.zeros((sim.n_envs, cfg.STATE_LAYOUT_CHANNELS, n_max_divs, n_max_divs), device=device)
        self.set_layout()

    # --------------------------------------------------------------------------
    # MARK: get_padded

    @staticmethod
    def get_padded_obj_pts(env: MazeEnv) -> np.ndarray:
        obj_pts = np.full((cfg.N_GOAL_CLRS, 2), np.inf)
        obj_pts[env.data.obj_goal_map] = env.data.obj_pts

        return obj_pts[None].repeat(env.sampler.n_bots, axis=0)

    @staticmethod
    def get_padded_wall_grid(env: MazeEnv, n_max_divs: int) -> np.ndarray:
        n0 = (n_max_divs - env.sampler.n_side_divs) // 2
        n1 = n_max_divs - env.sampler.n_side_divs - n0

        return np.pad(env.data.cell_wall_grid, [(n0, n1)]*2 + [(0, 0)]*3)

    @staticmethod
    def get_padded_clr_grid(env: MazeEnv, n_max_divs: int) -> np.ndarray:
        n0 = (n_max_divs - env.sampler.n_side_divs) // 2
        n1 = n_max_divs - env.sampler.n_side_divs - n0

        return np.pad(env.data.cell_clr_idx_grid, ((n0, n1), (n0, n1)), constant_values=-1)

    # --------------------------------------------------------------------------
    # MARK: set_layout

    def set_layout(self, env_idcs: 'list[int]' = None):
        """Mark the cells with objects for given envs."""

        obj_idcs = torch.arange(cfg.N_GOAL_CLRS)
        obj_cell_idcs = torch.from_numpy(self.obj_cell_idcs[:, [env.bot_idx for env in self.sim.envs]])

        for i in range(self.sim.n_envs) if env_idcs is None else env_idcs:
            self.cell_layout[(i, obj_idcs, *obj_cell_idcs[:, i].unbind(1))] = 1.
            self.cell_layout[i, cfg.N_GOAL_CLRS:-1] = self.cell_wall_map[i]
            self.cell_layout[i, -1] = self.cell_seg_map[i]

    # --------------------------------------------------------------------------
    # MARK: reset_tensors

    def reset_tensors(self):
        """Restore tensor states after an env. reset."""

        if not self.env_rst_idcs:
            return

        rst_envs = [self.sim.envs[i] for i in self.env_rst_idcs]
        bot_rst_idcs = torch.nonzero(self.bot_rst_mask).flatten()
        bot_rst_idcs_arr = bot_rst_idcs.cpu().numpy()

        if len(self.env_rst_idcs) > 1:
            bot_rst_subs = bot_rst_idcs

        else:
            bot_rst_subs = rst_envs[0].bot_slice

        # Bots
        bot_pos = np.concatenate([env.data.spawn_pts for env in rst_envs])
        self.bot_pos[bot_rst_subs] = bot_pos = torch.from_numpy(bot_pos).to(self.device, dtype=torch.float32)
        self.bot_vel[bot_rst_subs] = 0.

        # Goals/objects
        obj_pos = np.concatenate([self.get_padded_obj_pts(env) for env in rst_envs])
        self.obj_pos[bot_rst_subs] = obj_pos = torch.from_numpy(obj_pos).to(self.device, dtype=torch.float32)

        goal_idx_arr = np.concatenate([env.data.bot_goal_map for env in rst_envs])
        self.goal_idx_arr[bot_rst_idcs_arr] = goal_idx_arr

        self.goal_idx[bot_rst_subs] = goal_idx = torch.from_numpy(goal_idx_arr).to(self.device, dtype=torch.int64)
        self.goal_pos[bot_rst_subs] = goal_pos = self.obj_pos[bot_rst_idcs, goal_idx]
        self.goal_mask_f[bot_rst_subs] = one_hot(goal_idx, cfg.N_GOAL_CLRS).float()

        self.obj_pos_arr[:, bot_rst_idcs_arr] = np.moveaxis(obj_pos.cpu().numpy(), 1, 0)
        self.goal_pos_arr[bot_rst_idcs_arr] = goal_pos.cpu().numpy()

        self.obj_cell_idcs[:, bot_rst_idcs_arr] = np.concatenate([
            np.digitize(self.obj_pos_arr[:, env.bot_slice], env.sampler.open_delims) for env in rst_envs], axis=1)

        # Task tracking
        for env in rst_envs:
            self.scores[env.idx] = self.bot_done_mask_f[env.bot_slice].mean()

        if self.time_table is not None:
            self.time_table = torch.cat((self.time_table, torch.full_like(self.time_table[:1], -1)), dim=0)

        self.cell_rwd_sum[bot_rst_subs] = 0.
        self.goal_pred_ok[bot_rst_subs] = 0.
        self.goal_path_delta[bot_rst_subs] = 0.
        self.goal_path_len[bot_rst_subs] = 0.
        self.obj_in_frame[bot_rst_subs] = False
        self.obj_found[self.env_rst_idcs] = False

        self.bot_done_mask[bot_rst_subs] = False
        self.bot_done_mask_f[bot_rst_subs] = 0.
        self.env_step_ctrs[self.env_rst_idcs] = 0

        if self.ep_duration is None:
            for env in rst_envs:
                self.ep_durations[env.idx] = env.data.ep_duration

        else:
            self.ep_durations[self.env_rst_idcs] = self.ep_duration

        self.env_max_steps = (self.ep_durations * self.steps_per_second).long()

        # Halved duration for envs. with global spawns
        global_spawn_idcs = [i for i in self.env_rst_idcs if self.sim.envs[i].data.global_spawn_flag]

        if global_spawn_idcs:
            self.env_step_ctrs[global_spawn_idcs] = self.env_max_steps[global_spawn_idcs] // 2

        # Action feedback
        self.actions[bot_rst_subs] = 0.
        self.act_trq[bot_rst_subs] = 0.
        self.act_com[bot_rst_subs] = 0.

        self.speaker_mask_f[bot_rst_subs] = torch.from_numpy(
            np.concatenate([env.data.speaker_mask for env in rst_envs])).to(self.device, dtype=torch.float32)

        # Belief
        self.obj_in_mind[bot_rst_subs] = False
        self.goal_pos_in_mind[bot_rst_subs] = bot_pos

        # Grid cell states
        self.bot_cell_idcs[bot_rst_subs] = bot_cell_idcs = torch.bucketize(bot_pos, self.max_delims)

        self.cell_bot_map[bot_rst_subs] = 0.
        self.cell_bot_map[(bot_rst_idcs, *bot_cell_idcs.unbind(1))] = 1.
        self.cell_found_map[bot_rst_subs] = self.cell_bot_map[bot_rst_subs]

        n_max_divs = len(self.max_delims) + 1

        cell_wall_grid = np.stack([self.get_padded_wall_grid(env, n_max_divs) for env in rst_envs])
        cell_wall_mask = np.any(cell_wall_grid[..., 0, :] != cell_wall_grid[..., 1, :], axis=-1)
        cell_wall_map = np.stack((
            cell_wall_mask[:, :n_max_divs, :n_max_divs, cfg.SIDE_N_IDX],
            cell_wall_mask[:, :n_max_divs, :n_max_divs, cfg.SIDE_W_IDX],
            cell_wall_mask[:, 1:n_max_divs+1, :n_max_divs, cfg.SIDE_N_IDX],
            cell_wall_mask[:, :n_max_divs, 1:n_max_divs+1, cfg.SIDE_W_IDX]), axis=1)

        cell_cidx_map = np.stack([self.get_padded_clr_grid(env, n_max_divs) for env in rst_envs])

        self.cell_wall_map[self.env_rst_idcs] = torch.from_numpy(cell_wall_map).to(torch.float32)
        self.cell_seg_map[self.env_rst_idcs] = torch.from_numpy(cell_cidx_map).to(self.device, dtype=torch.float32)

        self.cell_layout[self.env_rst_idcs] = 0.
        self.set_layout(self.env_rst_idcs)

    # --------------------------------------------------------------------------
    # MARK: reset_sim

    def reset_sim(self):
        """Reinit. envs. and update the associated root states."""

        if not self.env_rst_idcs:
            return

        # Get updated states
        # NOTE: Partial reset implemented, but unused
        actor_states, actor_idcs = self.sim.reset(self.env_rst_idcs, True)

        # Set new states
        gym_idcs = torch.from_numpy(actor_idcs).to(self.device, dtype=torch.int32)
        self.actor_states[actor_idcs] = torch.from_numpy(actor_states).to(self.device, dtype=torch.float32)

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim.handle,
            gymtorch.unwrap_tensor(self.actor_states),
            gymtorch.unwrap_tensor(gym_idcs),
            len(actor_idcs))

        # Reset viewer perspective
        if not self.headless:
            self.interface.reset()

    def reset_all(self):
        self.bot_rst_mask.fill_(True)
        self.env_rst_idcs = list(range(self.sim.n_envs))

    # --------------------------------------------------------------------------
    # MARK: step

    def step(
        self,
        actions: Tensor = None,
        beliefs: Tensor = None,
        get_info: bool = True
    ) -> 'tuple[tuple[Tensor, ...], dict[str, Tensor | tuple[Tensor, ...]], dict[str, float]]':
        """
        Apply actions in environments, evaluate their effects,
        step physics and graphics, gather observations and other data.
        """

        if self.headless:
            self.apply_actions(actions, beliefs)

        else:
            while self.interface.paused:
                self.interface.eval_events()
                self.interface.sync_redraw(after_eval=False)

            # Get events
            self.interface.eval_events()

            # Pre-physics step
            self.apply_actions(actions, beliefs)

            # Step physics and graphics (the last pass in stride is made later)
            for _ in range(self.inference_stride-1):
                self.gym.simulate(self.sim.handle)
                self.gym.step_graphics(self.sim.handle)

                self.gym.fetch_results(self.sim.handle, True)
                self.interface.sync_redraw(after_eval=False)

        # Reset environments
        # NOTE: Resets are one step delayed to ensure that setter fns.,
        # followed by a sim. step, take effect before getting new observations,
        # at the cost of viewing the final actions taken in flagged environments
        # as arbitrary (which they generally are)
        self.reset_sim()

        # Async update tensors and colours, step physics
        self.async_event_loop.run_until_complete(asyncio.gather(
            self.async_reset_and_recolour(),
            self.async_simulate()))

        nrst_mask_f = (~self.bot_rst_mask.unsqueeze(-1)).float()

        # Async compute observations and rewards from physical state
        # NOTE: Refreshing cameras is the longest part of a step
        self.async_event_loop.run_until_complete(asyncio.gather(
            self.async_eval_state(),
            self.async_step_graphics()))

        obs_vec, obs_map, aux_val, rwd, prio_event_mask = self.async_temp_result

        obs_img = self.get_image_observations()
        obs = obs_img, obs_vec, obs_map

        # Externally handled log
        log_info = self.get_metrics() if get_info else self.NULL_INFO

        step_data = {
            'rwd': rwd,
            'prio': prio_event_mask,
            'vaux': aux_val,
            'nrst': nrst_mask_f}

        return obs, step_data, log_info

    # --------------------------------------------------------------------------
    # MARK: get_metrics
    def get_metrics(self) -> 'dict[str, float]':
        """Get the average number of tasks completed per env. and other metrics."""

        metrics = {
                'score': sum(self.scores) / self.sim.n_envs,
                'speed': torch.linalg.norm(self.bot_vel, dim=-1).mean().item(),
                'verbosity': self.act_com.mean().item()}

        for lvl, n_envs, slc in self.sim.level_slices:
            metrics[f'score{lvl}'] = sum(self.scores[slc]) / n_envs

        return metrics

    # --------------------------------------------------------------------------
    # MARK: apply_actions

    def apply_actions(self, actions: Tensor, beliefs: Tensor):
        """Set torques, relay beliefs."""

        if actions is None:
            return

        self.act_trq, self.act_com = actions.split(cfg.ACT_SPLIT, dim=1)

        trq_noise_scale = cfg.IO_NOISE_SCALE * 1.1
        self.act_trq += torch.where(self.act_trq != 0., torch.empty_like(self.act_trq).normal_(std=trq_noise_scale), 0.)

        # self.act_trq = torch.clamp(self.act_trq, -cfg.MOT_MAX_TORQUE, cfg.MOT_MAX_TORQUE)
        self.act_com = torch.clamp(self.act_com * self.speaker_mask_f.unsqueeze(-1), 0., 1.)

        act_trq = self.act_trq.reshape(-1)
        self.gym.set_dof_actuation_force_tensor(self.sim.handle, gymtorch.unwrap_tensor(act_trq))

        if beliefs is None:
            return

        obj_prob_in_mind, self.goal_pos_in_mind = beliefs.split(cfg.BELIEF_SPLIT, dim=1)

        self.obj_in_mind = obj_prob_in_mind > cfg.MIN_CLR_CLS_CONFIDENCE
        self.goal_pos_in_mind *= cfg.MAX_COORD_VAL

    # --------------------------------------------------------------------------
    # MARK: async_reset_and_recolour

    async def async_reset_and_recolour(self):
        self.reset_tensors()

    # --------------------------------------------------------------------------
    # MARK: async_simulate

    async def async_simulate(self):
        self.gym.simulate(self.sim.handle)

    # --------------------------------------------------------------------------
    # MARK: async_eval_state

    async def async_eval_state(self):
        """Get state data, compute rewards, check terminal conditions."""

        sim = self.sim

        # Update tensor data
        self.gym.fetch_results(sim.handle, True)
        self.gym.refresh_actor_root_state_tensor(sim.handle)
        self.gym.refresh_net_contact_force_tensor(sim.handle)

        # Get physical states
        self.bot_states = self.actor_states[self.all_bot_idcs]

        self.bot_pos = self.bot_states[:, :2]
        self.bot_pos_arr = bot_pos_arr = self.bot_pos.cpu().numpy()

        bot_ori = self.bot_states[:, 3:7]
        loc_ori = bot_ori * self.quat_inv

        obj_diff, obj_dist, obj_in_clear_arr = self.get_obj_relations(bot_pos_arr, loc_ori)
        min_bot_dist, near_src_prox, prox_sig_agg = self.get_prox_data(bot_ori)
        acc, ang_vel, sin_cos_z = self.get_imu_data(bot_ori, loc_ori)

        # Evaluate task progress
        new_bot_done, goal_dir, goal_path_dir, goal_dist, obj_masked_path_len = \
            self.eval_bot_tasks(bot_pos_arr, obj_in_clear_arr, loc_ori)

        # Check terminal conditions (all tasks completed or out of time)
        self.eval_env_reset()

        # Compute rewards based on task progress and physical state
        prio_event_mask, rwd, goal_found, colliding = self.eval_rewards(new_bot_done, near_src_prox)

        # Assemble observations
        cell_bot_map = torch.stack([self.cell_bot_map[env.bot_slice].sum(0).add_(1.).log_() for env in sim.envs])
        cell_found_map = torch.stack([self.cell_found_map[env.bot_slice].sum(0).add_(1.).log_() for env in sim.envs])

        # Ex(L+2)xHxW
        obs_map = torch.cat((self.cell_layout, cell_bot_map.unsqueeze(1), cell_found_map.unsqueeze(1)), dim=1)

        # Avg. min. dist. & vel. norm as crowding, congestion indicators (2)
        bot_stats = torch.stack((min_bot_dist.add_(1.).log_(), torch.linalg.norm(self.bot_vel, dim=-1)), dim=-1)
        bot_stats = torch.stack([bot_stats[env.bot_slice].mean(0) for env in sim.envs])

        # Avg. path and tasks remaining as progress per obj. (2*8)
        obj_stats = torch.cat((self.goal_mask_f, obj_masked_path_len), dim=-1)
        obj_stats = torch.stack([obj_stats[env.bot_slice].sum(0) / env.sampler.n_bots_per_goal for env in sim.envs])

        # Min. path remaining per obj. (8)
        obj_path_len = torch.where(self.goal_mask_f == 0., torch.inf, obj_masked_path_len)
        obj_path_len = torch.stack([obj_path_len[env.bot_slice].min(dim=0)[0] for env in sim.envs])
        obj_path_len = torch.where(obj_path_len == torch.inf, 0., obj_path_len)

        # Pooled over env. (34 = 2+16+8+8)
        env_stats = torch.cat((
            bot_stats,
            obj_stats,
            obj_path_len,
            self.obj_found.float()
        ), dim=-1).repeat_interleave(self.all_n_bots, dim=0, output_size=sim.n_all_bots)

        # Mask values for uninitialised goals
        obj_pos = torch.where(self.obj_pos == torch.inf, 2.*cfg.MAX_COORD_VAL, self.obj_pos)
        obj_dist[obj_dist == torch.inf] = 0.

        # Expose coordinates to oracle models
        goal_pos = self.goal_pos / cfg.MAX_COORD_VAL
        goal_found = goal_found.unsqueeze(-1)

        if self.oracular:
            aux_pos = goal_pos * (goal_found & ~self.bot_done_mask.unsqueeze(-1))

        else:
            aux_pos = self.zero_column.expand(-1, 2)

        # Add noise to obs. not already displaying noisy characteristics
        pos_noise_scale = cfg.IO_NOISE_SCALE / 10.
        bot_pos = (self.bot_pos / cfg.MAX_COORD_VAL).add_(torch.empty_like(self.bot_pos).normal_(std=pos_noise_scale))
        sin_cos_z += torch.empty_like(sin_cos_z).normal_(std=cfg.IO_NOISE_SCALE)
        prox_sig_agg = prox_sig_agg.add_(torch.empty_like(prox_sig_agg).normal_(std=cfg.IO_NOISE_SCALE)).clip_(0.)

        obs_vec = torch.cat((
            # Main vec. obs. (28 = 5+4+3+2+4+10); for the actors
            # DOF, light switch (4+1)
            self.act_trq,
            self.act_com,
            # GPS XY, DXY (2+2)
            bot_pos,
            self.bot_vel,
            # IMU ACC XY, ROT Z (2+1)
            acc,
            ang_vel,
            # AHRS SIN, COS (2)
            sin_cos_z,
            # Prox. channels (4)
            prox_sig_agg,
            # Task specification (8+2)
            self.goal_mask_f,
            aux_pos,
            # Hidden state (80 = 50+16+9+5); only for the critic
            # Bot & obj. stats (34+2*8)
            env_stats,
            obj_pos.flatten(1) / cfg.MAX_COORD_VAL,
            # Obj. relations (8+8)
            obj_dist / cfg.MAX_GOAL_DIST,
            self.obj_in_frame,
            # Task state & progress (2+3+3+1)
            goal_pos,
            goal_dir,
            goal_dist.unsqueeze(-1) / cfg.MAX_GOAL_DIST,
            goal_path_dir,
            self.goal_path_len.unsqueeze(-1) / cfg.MAX_GOAL_DIST,
            goal_found,
            # Rwd. components (5)
            self.goal_pred_ok.unsqueeze(-1),
            self.goal_path_delta.unsqueeze(-1).clamp(-cfg.CELL_DIAG_LENGTH, cfg.CELL_DIAG_LENGTH),
            self.cell_rwd_sum.unsqueeze(-1) / cfg.MAX_GOAL_REACHED_RWD,
            near_src_prox.unsqueeze(-1),
            colliding.unsqueeze(-1)
        ), dim=-1)

        # Make one-hot aux. val. target (1+8+2)
        aux_val = torch.cat((~self.obj_in_frame.any(-1, keepdim=True), obs_vec[:, cfg.AUX_VAL_SLICE]), dim=-1)

        self.async_temp_result = (obs_vec, obs_map, aux_val, rwd, prio_event_mask)

    # --------------------------------------------------------------------------
    # MARK: get_obj_relations

    def get_obj_relations(self, bot_pos_arr: np.ndarray, loc_ori: Tensor) -> 'tuple[Tensor, ...]':
        """Get sight masks, direction, and distance to each objective."""

        # Check line of sight
        obj_in_clear_arr = np.stack([
            np.concatenate([
                eval_line_of_sight(
                    bot_pos_arr[env.bot_slice],
                    obj_pos_arr[env.bot_slice],
                    obj_cell_idcs[env.bot_slice],
                    env.sampler.open_delims,
                    env.data.cell_wall_grid)
                for env in self.sim.envs])
            for obj_pos_arr, obj_cell_idcs in zip(self.obj_pos_arr, self.obj_cell_idcs)], axis=-1)

        obj_in_clear = torch.from_numpy(obj_in_clear_arr).to(self.device)

        # NxGx2, Nx2 -> NxGx2 - Nx1x2 -> NxGx2
        obj_diff = self.obj_pos - self.bot_pos.unsqueeze(1)

        # NxGx2 -> NxG
        obj_dist = torch.linalg.norm(obj_diff, dim=-1)

        # NxGx2 -> NxGx3
        zero_padding = self.zero_column.unsqueeze(1).expand(-1, cfg.N_GOAL_CLRS, -1)
        obj_diff3 = torch.cat((obj_diff, zero_padding), dim=-1)

        # Nx4, NxGx3 -> (N*G)x4, (N*G)x3
        loc_ori = loc_ori.repeat_interleave(cfg.N_GOAL_CLRS, dim=0, output_size=self.sim.n_all_bots * cfg.N_GOAL_CLRS)
        obj_diff3 = obj_diff3.reshape(-1, 3)

        # (N*G)x4, (N*G)x3 -> (N*G)x3
        loc_diff3 = apply_quat_rot(loc_ori, obj_diff3)

        # (N*G)x3 -> N*G -> NxG
        obj_in_fov = check_fov(loc_diff3).reshape(self.sim.n_all_bots, cfg.N_GOAL_CLRS)
        self.obj_in_frame = obj_in_clear & obj_in_fov

        return obj_diff, obj_dist, obj_in_clear_arr

    # --------------------------------------------------------------------------
    # MARK: get_prox_data

    def get_prox_data(self, bot_ori: Tensor) -> 'tuple[Tensor, ...]':
        """
        Detect near signal strength (proximity for entities in clipped range)
        wrt. 4 oriented receivers, each covering an angle of 90 degrees.
        """

        min_bot_dists = []
        near_src_proxs = []
        prox_sig_aggs = []

        for env in self.sim.envs:

            # Pairwise distances
            # Nx2 -> Bx2 -> 1xBx2 - Bx1x2 -> BxBx2
            bot_pos = self.bot_pos[env.bot_slice]
            bot_diff = bot_pos.unsqueeze(0) - bot_pos.unsqueeze(1)

            # Suppress own signal in distance calculation
            bot_dist = torch.linalg.norm(bot_diff, dim=-1)
            bot_dist[bot_dist == 0.] = torch.inf

            # Proximity weights via truncated inverse prop. characteristic
            sig_strength = norm_distance(bot_dist, cfg.PROX_SIGNAL_RADIUS).mul_(2.)

            # Angle-range weights
            # NOTE: Own angles are arbitrary, but their weight should already be suppressed
            incoming_angle = torch.atan2(bot_diff[..., 1], bot_diff[..., 0])
            rcvr_angles = clip_angle_range(get_eulz_from_quat(bot_ori[env.bot_slice]) + self.rcvr_rel_phases)

            # Bx(B+G), Bx4 -> Bx(B+G)x1 - Bx1x4 -> Bx(B+G)x4
            rel_angles = incoming_angle.unsqueeze(-1) - rcvr_angles.unsqueeze(1)
            rel_angles = clip_angle_range(rel_angles)

            # Phase-shifted filters
            rcvr_weights = rel_angles.cos_().clip_(0.)

            # Combined weights aggregated over sig. srcs.
            # Bx(B+G)x4, Bx(BxG) -> Bx(B+G)x4 -> Bx4
            prox_sig_agg = rcvr_weights.mul_(sig_strength.unsqueeze(-1)).sum(1)

            # BxB -> B
            min_bot_dist = bot_dist.min(dim=-1)[0]

            # Bx(B+G) -> B
            near_src_prox = (1. - bot_dist / (2.*cfg.BOT_WIDTH)).clip(0.).sum(-1)

            min_bot_dists.append(min_bot_dist)
            near_src_proxs.append(near_src_prox)
            prox_sig_aggs.append(prox_sig_agg)

        min_bot_dist = torch.cat(min_bot_dists)
        near_src_prox = torch.cat(near_src_proxs)
        prox_sig_agg = torch.cat(prox_sig_aggs)

        return min_bot_dist, near_src_prox, prox_sig_agg

    # --------------------------------------------------------------------------
    # MARK: get_imu_data

    def get_imu_data(self, bot_ori: Tensor, loc_ori: Tensor) -> 'tuple[Tensor, ...]':
        """Simulate the accelerometer, gyroscope, and AHRS."""

        bot_vel = self.bot_states[:, 7:9]

        # Simulate accelerometer
        acc = (bot_vel - self.bot_vel) / self.dt
        acc = apply_quat_rot(loc_ori, torch.cat((acc, self.zero_column), dim=-1))[:, :2]

        # Expecting at most |0. - 0.3m/0.25s| / 0.25s at abrupt stop from full speed
        acc /= 5.

        self.bot_vel = bot_vel

        # Expecting at most one turn per 6 seconds
        ang_vel = self.bot_states[:, -1:]

        # AHRS
        sin_cos_z = get_trigonz_from_quat(bot_ori)

        return acc, ang_vel, sin_cos_z

    # --------------------------------------------------------------------------
    # MARK: eval_bot_tasks

    def eval_bot_tasks(self, bot_pos_arr: np.ndarray, obj_in_clear_arr: Tensor, loc_ori: Tensor) -> 'tuple[Tensor,...]':
        """Check if any goals are in reach and confirm completed tasks."""

        goal_in_clear_arr = obj_in_clear_arr[self.row_idx_arr, self.goal_idx_arr]

        # Get A* path length and current direction
        env_path_est = [
            env.get_path_estimates(
                bot_pos_arr[env.bot_slice],
                self.goal_pos_arr[env.bot_slice],
                goal_in_clear_arr[env.bot_slice])
            for env in self.sim.envs]

        goal_path_len = np.concatenate([path_est[0] for path_est in env_path_est])
        goal_path_len = torch.from_numpy(goal_path_len).to(self.device, dtype=torch.float32)

        goal_path_dir = np.concatenate([path_est[1] for path_est in env_path_est])
        goal_path_dir = torch.from_numpy(goal_path_dir).to(self.device, dtype=torch.float32)

        goal_in_clear = torch.from_numpy(goal_in_clear_arr).to(self.device)

        # Get air distance and direction
        goal_diff = self.goal_pos - self.bot_pos
        goal_dist = torch.linalg.norm(goal_diff, dim=1)

        # Rotate directions to local frame
        goal_diff = torch.cat((goal_diff, self.zero_column), dim=-1)
        goal_path_dir = torch.cat((goal_path_dir, self.zero_column), dim=-1)

        goal_diff = apply_quat_rot(loc_ori, goal_diff)[:, :2]
        goal_path_dir = apply_quat_rot(loc_ori, goal_path_dir)[:, :2]

        goal_dir = goal_diff / torch.linalg.norm(goal_diff, dim=1, keepdim=True)
        goal_path_dir = goal_path_dir / torch.linalg.norm(goal_path_dir, dim=1, keepdim=True)

        # Confirm goal reached by line of sight, proximity, and recognition
        goal_in_frame = goal_in_clear & check_fov(goal_diff)
        goal_in_reach = goal_in_frame & (goal_dist < cfg.GOAL_ZONE_RADIUS)

        goal_in_mind = self.obj_in_mind[self.row_idcs, self.goal_idx]
        goal_reached = goal_in_reach & goal_in_mind

        new_bot_done = (~self.bot_done_mask & goal_reached).float()

        # Update task trackers
        self.bot_done_mask |= goal_reached
        self.bot_done_mask_f = self.bot_done_mask.float()
        bot_undone_mask_f = 1. - self.bot_done_mask_f

        goal_path_len *= bot_undone_mask_f

        self.goal_path_delta = torch.where(
            self.goal_path_len.bool() & goal_path_len.bool(),
            self.goal_path_delta.add_(self.goal_path_len).sub_(goal_path_len),
            0.)

        self.goal_path_len = goal_path_len
        self.goal_mask_f *= bot_undone_mask_f.unsqueeze(-1)
        obj_masked_path_len = self.goal_mask_f * goal_path_len.unsqueeze(-1) / cfg.MAX_GOAL_DIST

        return new_bot_done, goal_dir, goal_path_dir, goal_dist, obj_masked_path_len

    # --------------------------------------------------------------------------
    # MARK: eval_env_reset

    def eval_env_reset(self):
        """Step time and check for terminal envs. ahead of next sim./eval. step."""

        self.env_step_ctrs += 1
        env_rst_mask = self.env_step_ctrs == self.env_max_steps

        self.env_rst_idcs = torch.nonzero(env_rst_mask).flatten().tolist()
        self.bot_rst_mask = env_rst_mask.repeat_interleave(self.all_n_bots, output_size=self.sim.n_all_bots)

    # --------------------------------------------------------------------------
    # MARK: eval_rewards

    def eval_rewards(self, new_bot_done: Tensor, near_src_prox: Tensor) -> 'tuple[Tensor, ...]':
        """Check conditions for joint and individual rewards."""

        sim = self.sim

        # Joint rewards
        if self.belief_gain:
            goal_pred_ok = torch.linalg.norm(self.goal_pos - self.goal_pos_in_mind, dim=-1) < cfg.GOAL_PRED_RADIUS
            goal_pred_ok = (goal_pred_ok | self.bot_done_mask).float()

            goal_pred_delta = (goal_pred_ok - self.goal_pred_ok).sign()
            goal_pred_delta = \
                torch.stack([goal_pred_delta[env.bot_slice].sum() / env.sampler.n_bots_per_goal for env in sim.envs])

            self.goal_pred_ok = goal_pred_ok

        else:
            goal_pred_delta = 0.

        # Event when a bot correctly recognises that a spec. obj. is in sight
        obj_found = self.obj_in_frame & self.obj_in_mind

        # NxG -> ExG -> E
        obj_found = torch.stack([obj_found[env.bot_slice].any(0) for env in sim.envs])
        new_obj_found = ~self.obj_found & obj_found
        any_obj_found = new_obj_found.any(1)

        if self.time_table is not None:
            if any_obj_found.any():
                goal_found_mask = new_obj_found.unsqueeze(1) * self.goal_mask_f.reshape(sim.n_envs, sim.n_bots, -1)
                goal_found_time = goal_found_mask.sum(-1) * (self.env_step_ctrs.unsqueeze(1) + 1)
                self.time_table[-1, ..., 0] += goal_found_time.long()

            if new_bot_done.any():
                goal_reached_time = new_bot_done.reshape(sim.n_envs, sim.n_bots) * (self.env_step_ctrs.unsqueeze(1) + 1)
                self.time_table[-1, ..., 1] += goal_reached_time.long()

        self.obj_found |= obj_found
        goal_found = self.obj_found.repeat_interleave(
            self.all_n_bots, dim=0, output_size=sim.n_all_bots)[self.row_idcs, self.goal_idx]

        # Mark envs. to put a short sequence of steps into the aux. task prio. buffer
        prio_event_mask = any_obj_found

        joint_rwd = any_obj_found * cfg.OBJ_FOUND_RWD + goal_pred_delta * cfg.BLIF_DELTA_RWD

        # Long-horizon indiv. rewards
        bot_cell_idcs = torch.bucketize(self.bot_pos, self.max_delims)
        bot_cell_idx_tuple = (self.row_idcs, *bot_cell_idcs.unbind(1))

        # Pos. before goal is found, signed after: pos. if new cell closer to goal, neg. if farther
        cell_found = 1. - self.cell_found_map[bot_cell_idx_tuple]

        if self.belief_util:
            signed_cell_found = torch.where(
                goal_found,
                (self.bot_cell_idcs != bot_cell_idcs).any(-1) * self.goal_path_delta.sign(),
                cell_found)

        else:
            signed_cell_found = cell_found

        new_cell_rwd = signed_cell_found * cfg.CELL_REACHED_RWD

        # Full remainder given on task completion
        indiv_rwd = torch.where(
            self.bot_done_mask,
            new_bot_done * (cfg.MAX_GOAL_REACHED_RWD - self.cell_rwd_sum),
            new_cell_rwd)

        self.goal_path_delta[signed_cell_found.bool()] = 0.
        self.cell_rwd_sum = torch.where(self.bot_done_mask, 0., self.cell_rwd_sum + new_cell_rwd)

        self.cell_bot_map.zero_()
        self.cell_bot_map[bot_cell_idx_tuple] = 1.
        self.cell_found_map[bot_cell_idx_tuple] = 1.
        self.bot_cell_idcs = bot_cell_idcs

        # Short-horizon indiv. penalties
        colliding = self.net_contacts[self.collider_idcs, :2].abs().sum(-1) != 0.

        indiv_pen = colliding * cfg.COLLISION_RWD + near_src_prox * cfg.PROXIMITY_RWD

        # Stack joint and individual rewards
        joint_rwd = joint_rwd.repeat_interleave(self.all_n_bots, output_size=sim.n_all_bots)
        rwd = torch.stack((joint_rwd, indiv_rwd, indiv_pen), dim=-1)

        return prio_event_mask, rwd, goal_found, colliding

    # --------------------------------------------------------------------------
    # MARK: async_step_graphics

    async def async_step_graphics(self):
        self.gym.step_graphics(self.sim.handle)

        if self.render_cameras:
            self.gym.render_all_camera_sensors(self.sim.handle)

        if not self.headless:
            self.interface.sync_redraw()

    # --------------------------------------------------------------------------
    # MARK: get_image_observations

    def get_image_observations(self) -> Tensor:
        if not self.render_cameras:
            return self.null_obs_img

        self.gym.start_access_image_tensors(self.sim.handle)

        rgb = torch.stack(self.img_rgb_list)[..., :3]
        dep = torch.stack(self.img_dep_list)
        seg = torch.stack(self.img_seg_list) if self.keep_segmentation else None

        self.gym.end_access_image_tensors(self.sim.handle)

        # Normalise
        rgb = rgb / 255.

        # Add noise
        if self.sim.text_mode == cfg.TEXT_SPLIT:
            rgb_ = rgb[:self.sim.n_all_bots//2]
            rgb_ += torch.empty_like(rgb_).normal_(std=0.02)
            rgb_.clip_(0., 1.)

            # Add noise and bins
            dep = (-dep).clip_(0., cfg.MAX_IMG_DEPTH)
            dep_clean = norm_distance(dep, cfg.MAX_IMG_DEPTH) if self.keep_segmentation else None

            dep_ = dep[:self.sim.n_all_bots//2]
            dep_ += torch.empty_like(dep_).normal_() * dep_**2 * 0.007
            dep_ = norm_distance(dep_, cfg.MAX_IMG_DEPTH)
            dep_ += torch.empty_like(dep_).normal_(std=0.02) * (torch.empty_like(dep_).uniform_() < 0.1)
            dep_ = dep_.clip_(0., 1.)

            for tl, th in zip(*self.depth_bins):
                dep_[(dep_ <= tl) & (dep_ > th)] = tl

            dep = torch.cat((dep_, dep_clean[:self.sim.n_all_bots//2]))

        elif self.sim.text_mode > cfg.TEXT_WARES:
            rgb += torch.empty_like(rgb).normal_(std=0.02)
            rgb = rgb.clip_(0., 1.)

            # Add noise and bins
            dep = (-dep).clip_(0., cfg.MAX_IMG_DEPTH)
            dep_clean = norm_distance(dep, cfg.MAX_IMG_DEPTH) if self.keep_segmentation else None
            dep += torch.empty_like(dep).normal_() * dep**2 * 0.007
            dep = norm_distance(dep, cfg.MAX_IMG_DEPTH)
            dep += torch.empty_like(dep).normal_(std=0.02) * (torch.empty_like(dep).uniform_() < 0.1)
            dep = dep.clip_(0., 1.)

            for tl, th in zip(*self.depth_bins):
                dep[(dep <= tl) & (dep > th)] = tl

        else:
            dep = dep_clean = norm_distance(-dep, cfg.MAX_IMG_DEPTH)

        # Convert to HSV space and 3D form
        hsv = rgb_to_hsv(rgb, stack_dim=1)
        h, s, v = hsv.unbind(1)

        h *= 2. * torch.pi
        hsin = h.sin()
        hcos = h.cos()

        m = (1. - torch.linalg.norm(1. - hsv[:, 1:], dim=1)).clip(0.)
        hsv = torch.stack((hcos * m, hsin * m, v - s), dim=1)

        # Stack channels
        if self.keep_segmentation:
            rgb = rgb.permute(0, 3, 1, 2)
            obs_img = torch.cat((hsv, dep.unsqueeze(1), dep_clean.unsqueeze(1), seg.unsqueeze(1), rgb), dim=1)

        else:
            obs_img = torch.cat((hsv, dep.unsqueeze(1)), dim=1)

        return obs_img
