"""Rules and state transitions"""

import asyncio
from typing import Callable

import numpy as np
from isaacgym import gymapi, gymtorch
import torch
from torch import Tensor
from torch.nn.functional import one_hot

import config as cfg
from sim import MazeSim
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
        ep_duration: float = cfg.EP_DURATION,
        steps_per_second: int = cfg.STEPS_PER_SECOND,
        frames_per_second: int = 64,
        render_cameras: bool = False,
        keep_segmentation: bool = False,
        keep_rgb_over_hsv: bool = False,
        stagger_env_resets: bool = False,
        reward_belief_gain: bool = False,
        reward_belief_util: bool = False,
        track_performance: bool = False,
        device: str = 'cuda'
    ):
        self.sim = sim
        self.gym = sim.gym
        self.interface = interface
        self.headless = interface is None
        self.render_cameras = render_cameras
        self.keep_segmentation = keep_segmentation
        self.keep_rgb_over_hsv = keep_rgb_over_hsv
        self.stagger_env_resets = stagger_env_resets
        self.belief_gain = reward_belief_gain
        self.belief_util = reward_belief_util
        self.device = torch.device(device)

        steps_per_second = min(steps_per_second, frames_per_second)
        frames_per_second = steps_per_second if self.headless else frames_per_second

        if frames_per_second % steps_per_second:
            raise ValueError(f'Mismatch between FPS ({frames_per_second}) and action frequency ({steps_per_second}).')

        self.inference_stride = frames_per_second // steps_per_second
        self.dt = 1. / steps_per_second
        self.steps_in_ep = ep_duration * steps_per_second
        self.ep_duration = ep_duration

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

        # Views of data with fixed memory address
        asr = self.actor_states.reshape(sim.n_envs, -1, 13)
        self.env_bot_pos3 = asr[:, -sim.n_bots:, :3]
        self.env_bot_pos = asr[:, -sim.n_bots:, :2]
        self.env_bot_ori = asr[:, -sim.n_bots:, 3:7]

        self.env_bot_vel = asr[:, -sim.n_bots:, 7:9]
        self.env_bot_ang_vel = asr[:, -sim.n_bots:, -1]

        # Not views any more, must be continuously copied into variable address
        self.bot_pos = self.env_bot_pos.reshape(-1, 2).contiguous()
        self.bot_vel = self.env_bot_vel.reshape(-1, 2).contiguous()

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
        self.sky_clr = torch.tensor(COLOURS['background'][cfg.SKY_CLR_IDX], dtype=torch.float32, device=device)
        self.row_idcs = torch.arange(sim.n_all_bots, device=device)
        self.row_idx_arr = np.arange(sim.n_all_bots)

        # Goals/objects (ExGx2 -> NxGx2)
        self.obj_pos = np.stack([env.obj_pts for env in sim.envs])
        self.obj_pos = np.repeat(self.obj_pos, sim.n_bots, axis=0)
        self.obj_pos = torch.from_numpy(self.obj_pos).to(device, dtype=torch.float32)

        # Initial tasks
        self.goal_idx_arr = np.concatenate([env.bot_obj_map for env in sim.envs])
        self.goal_idx = torch.from_numpy(self.goal_idx_arr).to(device, dtype=torch.int64)
        self.goal_pos = self.obj_pos[self.row_idcs, self.goal_idx]
        self.goal_mask_f = one_hot(self.goal_idx, cfg.N_GOAL_CLRS).float()

        # Must be on CPU for custom line of sight tracing and path finding
        self.bot_pos_arr = self.bot_pos.cpu().numpy()
        self.obj_pos_arr = np.moveaxis(self.obj_pos.cpu().numpy(), 1, 0)
        self.goal_pos_arr = self.goal_pos.cpu().numpy()
        self.obj_cell_idcs = np.digitize(self.obj_pos_arr, sim.sampler.open_delims)

        # Task tracking
        self.cell_rwd_sum = torch.zeros(sim.n_all_bots, device=device)
        self.goal_pred_ok = torch.zeros(sim.n_all_bots, device=device)
        self.goal_path_delta = torch.zeros(sim.n_all_bots, device=device)
        self.goal_path_len = torch.zeros(sim.n_all_bots, device=device)
        self.obj_in_frame = torch.zeros((sim.n_all_bots, sim.n_goals), dtype=torch.bool, device=device)
        self.obj_found = torch.zeros((sim.n_envs, sim.n_goals), dtype=torch.bool, device=device)

        self.bot_done_mask = torch.zeros(sim.n_all_bots, dtype=torch.bool, device=device)
        self.bot_done_mask_f = torch.zeros(sim.n_all_bots, device=device)

        self.bot_rst_mask = torch.zeros(sim.n_all_bots, dtype=torch.bool, device=device)
        self.env_step_ctrs = torch.zeros(sim.n_envs, dtype=torch.int64, device=device)
        self.env_rst_idcs: 'list[int]' = []
        self.scores = []
        self.time_table = (
            torch.zeros((1, sim.n_envs, sim.n_bots, 2), dtype=torch.int64, device=device)
            if track_performance else None)

        # Force premature first env. resets by starting their counters mid-episode
        # Having envs. at different stages helps to decorrelate experience in batches
        # NOTE: Until envs. reach a steady state, the learning rate should be low or 0.
        if self.stagger_env_resets:
            envs_per_step = sim.n_envs / self.steps_in_ep
            envs_done = 0

            for i in range(self.steps_in_ep):
                envs_to_reset = int((i+1) * envs_per_step) - envs_done

                if envs_to_reset:
                    self.env_step_ctrs[envs_done:envs_done + envs_to_reset] = i
                    envs_done += envs_to_reset

            self.env_step_ctrs -= self.env_step_ctrs[0].item()
            # self.env_step_ctrs = self.env_step_ctrs[torch.randperm(sim.n_envs)]

        # Halve the remaining duration for envs. with global spawns
        global_spawn_idcs = [i for i, env in enumerate(sim.envs) if env.global_spawn_flag]

        if global_spawn_idcs:
            self.env_step_ctrs[global_spawn_idcs] //= 2
            self.env_step_ctrs[global_spawn_idcs] += self.steps_in_ep // 2
            self.cell_rwd_sum.view(self.sim.n_envs, -1)[global_spawn_idcs] = cfg.MAX_GOAL_REACHED_RWD / 2.

        # Action feedback
        self.actions = torch.zeros((sim.n_all_bots, sum(cfg.ACT_SPLIT)), device=device)
        self.act_trq, self.act_led = self.actions.split(cfg.ACT_SPLIT, dim=-1)

        # Belief
        self.obj_in_mind = torch.zeros((sim.n_all_bots, cfg.N_GOAL_CLRS), dtype=torch.bool, device=device)
        self.goal_pos_in_mind = self.bot_pos

        # Grid cell states
        self.side_delims = torch.from_numpy(sim.sampler.open_delims).to(device, dtype=torch.float32)
        n_side_divs = len(self.side_delims) + 1

        self.bot_cell_idcs = torch.bucketize(self.bot_pos, self.side_delims)

        # Bot presence & exploration
        self.cell_bot_map = torch.zeros((sim.n_all_bots, n_side_divs, n_side_divs), device=device)
        self.cell_bot_map[(self.row_idcs, *self.bot_cell_idcs.unbind(1))] = 1.
        self.cell_found_map = self.cell_bot_map.clone()

        # NWSE walls
        self.cell_wall_grid = sim.data['cell_wall_grid']
        self.cell_wall_map = torch.zeros((cfg.N_CARDINALS, n_side_divs, n_side_divs), device=device)

        cell_wall_mask = np.any(self.cell_wall_grid[..., 0, :] != self.cell_wall_grid[..., 1, :], axis=-1)
        cell_wall_mask = torch.from_numpy(cell_wall_mask).to(torch.float32)

        self.cell_wall_map[0] = cell_wall_mask[:n_side_divs, :n_side_divs, cfg.SIDE_N_IDX]
        self.cell_wall_map[1] = cell_wall_mask[:n_side_divs, :n_side_divs, cfg.SIDE_W_IDX]
        self.cell_wall_map[2] = cell_wall_mask[1:n_side_divs+1, :n_side_divs, cfg.SIDE_N_IDX]
        self.cell_wall_map[3] = cell_wall_mask[:n_side_divs, 1:n_side_divs+1, cfg.SIDE_W_IDX]

        # Clr. cls. segmentation
        cell_cidx_map = torch.tensor(sim.data['cell_clr_idx_grid']).to(device, dtype=torch.float32)
        self.cell_seg_map = torch.where(cell_cidx_map >= 0., (cell_cidx_map + 1.) / cfg.N_WALL_CLRS, -1.)

        # Obj. presence
        self.cell_layout = torch.zeros((sim.n_envs, cfg.STATE_LAYOUT_CHANNELS, n_side_divs, n_side_divs), device=device)
        self.set_layout()

    # --------------------------------------------------------------------------
    # MARK: set_layout

    def set_layout(self, env_idcs: 'list[int]' = None):
        """Mark the cells with objects for given envs."""

        obj_idcs = torch.arange(len(self.obj_cell_idcs))
        obj_cell_idcs = torch.from_numpy(self.obj_cell_idcs[:, ::self.sim.n_bots])

        for i in range(self.sim.n_envs) if env_idcs is None else env_idcs:
            self.cell_layout[(i, obj_idcs, *obj_cell_idcs[:, i].unbind(1))] = 1.
            self.cell_layout[i, cfg.N_GOAL_CLRS:-1] = self.cell_wall_map
            self.cell_layout[i, -1] = self.cell_seg_map

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
            bot_rst_subs = slice(self.env_rst_idcs[0]*self.sim.n_bots, (self.env_rst_idcs[0]+1)*self.sim.n_bots)

        # Bots
        bot_pos = np.concatenate([env.spawn_pts for env in rst_envs])
        self.bot_pos[bot_rst_subs] = bot_pos = torch.from_numpy(bot_pos).to(self.device, dtype=torch.float32)
        self.bot_vel[bot_rst_subs] = 0.

        # Goals/objects
        obj_pos = np.stack([env.obj_pts for env in rst_envs])
        obj_pos = np.repeat(obj_pos, self.sim.n_bots, axis=0)
        self.obj_pos[bot_rst_subs] = obj_pos = torch.from_numpy(obj_pos).to(self.device, dtype=torch.float32)

        goal_idx_arr = np.concatenate([env.bot_obj_map for env in rst_envs])
        self.goal_idx_arr[bot_rst_idcs_arr] = goal_idx_arr

        self.goal_idx[bot_rst_subs] = goal_idx = torch.from_numpy(goal_idx_arr).to(self.device, dtype=torch.int64)
        self.goal_pos[bot_rst_subs] = goal_pos = self.obj_pos[bot_rst_idcs, goal_idx]
        self.goal_mask_f[bot_rst_subs] = one_hot(goal_idx, cfg.N_GOAL_CLRS).float()

        self.obj_pos_arr[:, bot_rst_idcs_arr] = obj_pos_arr = np.moveaxis(obj_pos.cpu().numpy(), 1, 0)
        self.goal_pos_arr[bot_rst_idcs_arr] = goal_pos.cpu().numpy()

        self.obj_cell_idcs[:, bot_rst_idcs_arr] = np.digitize(obj_pos_arr, self.sim.sampler.open_delims)

        # Task tracking
        final_scores = self.bot_done_mask_f.reshape(self.sim.n_envs, self.sim.n_bots).mean(1)[self.env_rst_idcs]
        self.scores = self.scores[self.sim.n_envs - len(self.env_rst_idcs):] + final_scores.tolist()

        if self.time_table is not None:
            self.time_table = torch.cat((self.time_table, torch.zeros_like(self.time_table[:1])), dim=0)

        self.cell_rwd_sum[bot_rst_subs] = 0.
        self.goal_pred_ok[bot_rst_subs] = 0.
        self.goal_path_delta[bot_rst_subs] = 0.
        self.goal_path_len[bot_rst_subs] = 0.
        self.obj_in_frame[bot_rst_subs] = False
        self.obj_found[self.env_rst_idcs] = False

        self.bot_done_mask[bot_rst_subs] = False
        self.bot_done_mask_f[bot_rst_subs] = 0.
        self.env_step_ctrs[self.env_rst_idcs] = 0

        global_spawn_idcs = [i for i in self.env_rst_idcs if self.sim.envs[i].global_spawn_flag]

        if global_spawn_idcs:
            self.env_step_ctrs[global_spawn_idcs] = self.steps_in_ep // 2
            self.cell_rwd_sum.view(self.sim.n_envs, -1)[global_spawn_idcs] = cfg.MAX_GOAL_REACHED_RWD / 2.

        # Action feedback
        self.actions[bot_rst_subs] = 0.
        self.act_trq[bot_rst_subs] = 0.
        self.act_led[bot_rst_subs] = 0.

        # Belief
        self.obj_in_mind[bot_rst_subs] = False
        self.goal_pos_in_mind[bot_rst_subs] = bot_pos

        # Grid cell states
        self.bot_cell_idcs[bot_rst_subs] = bot_cell_idcs = torch.bucketize(bot_pos, self.side_delims)

        self.cell_bot_map[bot_rst_subs] = 0.
        self.cell_bot_map[(bot_rst_idcs, *bot_cell_idcs.unbind(1))] = 1.
        self.cell_found_map[bot_rst_subs] = self.cell_bot_map[bot_rst_subs]

        self.cell_layout[self.env_rst_idcs] = 0.
        self.set_layout(self.env_rst_idcs)

    # --------------------------------------------------------------------------
    # MARK: reset_sim

    def reset_sim(self):
        """Reinit. envs. and update the associated root states."""

        if not self.env_rst_idcs:
            return

        # Get updated states
        actor_states, actor_idcs = self.sim.reset(self.env_rst_idcs)

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
            # Pre-physics step
            self.apply_actions(actions, beliefs)

        else:
            while self.interface.paused:
                self.interface.eval_events()
                self.interface.sync_redraw(after_eval=False)

            # Get events
            self.interface.eval_events()
            self.apply_actions(actions, beliefs)

            # Step physics and graphics (the last stride is made later)
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
        self.run_async_ops(self.async_reset_and_recolour, self.async_simulate)
        nrst_mask_f = (~self.bot_rst_mask.unsqueeze(-1)).float()

        # Async compute observations and rewards from physical state
        # NOTE: Refreshing cameras is by far the longest part of a step
        self.run_async_ops(self.async_eval_state, self.async_step_graphics)
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

        return {
                'score': sum(self.scores) / max(1, len(self.scores)),
                'speed': torch.linalg.norm(self.bot_vel, dim=-1).mean().item(),
                'verbosity': self.act_led.mean().item()}

    # --------------------------------------------------------------------------
    # MARK: apply_actions

    def apply_actions(self, actions: Tensor, beliefs: Tensor):
        """Set torques, relay beliefs."""

        if actions is None:
            return

        self.act_trq, self.act_led = actions.split(cfg.ACT_SPLIT, dim=1)

        # NOTE: Default is short-range (white) signal, not silence (black)
        self.act_trq = torch.clamp(self.act_trq, -cfg.MOT_MAX_TORQUE, cfg.MOT_MAX_TORQUE)
        self.act_led = torch.clamp(self.act_led, 0., 1.)

        act_trq = self.act_trq.reshape(-1)
        self.gym.set_dof_actuation_force_tensor(self.sim.handle, gymtorch.unwrap_tensor(act_trq))

        if beliefs is None:
            return

        obj_prob_in_mind, self.goal_pos_in_mind = beliefs.split(cfg.BELIEF_SPLIT, dim=1)

        self.obj_in_mind = obj_prob_in_mind > 0.66
        self.goal_pos_in_mind *= cfg.MAX_COORD_VAL

    # --------------------------------------------------------------------------
    # MARK: async_reset_and_recolour

    async def async_reset_and_recolour(self):
        self.reset_tensors()

        # Colouring action
        clrs = self.act_led.expand(-1, 3).cpu().numpy()

        for (env_handle, bot_handle), rgb in zip(self.sim.env_bot_handles, clrs):
            self.gym.set_rigid_body_color(
                env_handle, bot_handle, cfg.BOT_BEACON_IDX, gymapi.MESH_VISUAL, gymapi.Vec3(*rgb))

    # --------------------------------------------------------------------------
    # MARK: async_simulate

    async def async_simulate(self, other_task: asyncio.Task):
        self.gym.simulate(self.sim.handle)

        await other_task
        self.async_event_loop.stop()

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
        self.bot_pos = self.env_bot_pos.reshape(-1, 2).contiguous()
        self.bot_pos_arr = bot_pos_arr = self.bot_pos.cpu().numpy()

        bot_ori = self.env_bot_ori.reshape(-1, 4)
        loc_ori = bot_ori * self.quat_inv

        obj_diff, obj_dist, obj_in_clear_arr = self.get_obj_relations(bot_pos_arr, loc_ori)
        bot_dist, src_dist, prox_sig_agg = self.get_prox_data(obj_diff, obj_dist, bot_ori)
        acc, ang_vel, sin_cos_z = self.get_imu_data(bot_ori, loc_ori)

        # Evaluate task progress
        new_bot_done, goal_dir, goal_path_dir, goal_dist, obj_masked_path_len, obj_undone_mask_f = \
            self.eval_bot_tasks(bot_pos_arr, obj_in_clear_arr, loc_ori)

        # Check terminal conditions (all tasks completed or out of time)
        time_left = self.eval_env_reset()

        # Compute rewards based on task progress and physical state
        prio_event_mask, rwd, goal_found, colliding, near_src_prox = self.eval_rewards(new_bot_done, src_dist)

        # Assemble observations
        cell_bot_map = self.cell_bot_map.reshape(sim.n_envs, sim.n_bots, *self.cell_bot_map.shape[1:])
        cell_bot_map = cell_bot_map.sum(1, keepdim=True).add_(1.).log_()

        cell_found_map = self.cell_found_map.reshape(sim.n_envs, sim.n_bots, *self.cell_found_map.shape[1:])
        cell_found_map = cell_found_map.sum(1, keepdim=True).add_(1.).log_()

        # Ex(L+2)xHxW
        obs_map = torch.cat((self.cell_layout, cell_bot_map, cell_found_map), dim=1)

        # Avg. min. dist. & vel. norm as crowding, congestion indicators (2)
        bot_stats = torch.stack((
            bot_dist.min(dim=-1)[0].add_(1.).log_(),
            torch.linalg.norm(self.bot_vel, dim=-1)
        ), dim=-1).reshape(sim.n_envs, sim.n_bots, -1).mean(1)

        # Avg. path and tasks remaining as progress per obj. (2*8)
        obj_stats = torch.cat((
            obj_undone_mask_f,
            obj_masked_path_len
        ), dim=-1).reshape(sim.n_envs, sim.n_bots, -1).sum(1) / sim.n_bots_per_goal

        # Min. path remaining per obj. (8)
        obj_path_len = torch.where(self.goal_mask_f == 0., torch.inf, obj_masked_path_len)
        obj_path_len = obj_path_len.reshape(sim.n_envs, sim.n_bots, -1).min(dim=1)[0]
        obj_path_len = torch.where(obj_path_len == torch.inf, 0., obj_path_len)

        # Pooled over env. (34 = 2+16+8+8)
        env_stats = torch.cat((
            bot_stats,
            obj_stats,
            obj_path_len,
            self.obj_found.float()
        ), dim=-1).repeat_interleave(sim.n_bots, dim=0, output_size=sim.n_all_bots)

        obs_vec = torch.cat((
            # Main vec. obs. (28 = 5+4+3+2+4+10); for the actors
            # DOF, light switch (4+1)
            self.act_trq,
            self.act_led,
            # GPS XY, DXY (2+2)
            self.bot_pos / cfg.MAX_COORD_VAL,
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
            self.bot_done_mask_f.unsqueeze(-1),
            time_left.unsqueeze(-1),
            # Hidden state (80 = 50+16+9+5); only for the critic
            # Bot & obj. stats (34+2*8)
            env_stats,
            self.obj_pos.flatten(1) / cfg.MAX_COORD_VAL,
            # Obj. relations (8+8)
            obj_dist / cfg.MAX_GOAL_DIST,
            self.obj_in_frame,
            # Task state & progress (2+3+3+1)
            self.goal_pos / cfg.MAX_COORD_VAL,
            goal_dir,
            goal_dist.unsqueeze(-1) / cfg.MAX_GOAL_DIST,
            goal_path_dir,
            self.goal_path_len.unsqueeze(-1) / cfg.MAX_GOAL_DIST,
            goal_found.unsqueeze(-1),
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
            eval_line_of_sight(
                bot_pos_arr,
                obj_pos_arr,
                obj_cell_idcs,
                self.sim.sampler.open_delims,
                self.cell_wall_grid)
            for obj_pos_arr, obj_cell_idcs in zip(self.obj_pos_arr, self.obj_cell_idcs)], axis=-1)

        obj_in_clear = torch.from_numpy(obj_in_clear_arr).to(self.device)

        # NxGx2, Nx2 -> NxGx2 - Nx1x2 -> NxGx2
        obj_diff = self.obj_pos - self.bot_pos.unsqueeze(1)

        # NxGx2 -> NxG
        obj_dist = torch.linalg.norm(obj_diff, dim=-1)

        # NxGx2 -> NxGx3
        zero_padding = self.zero_column.unsqueeze(1).expand(-1, self.sim.n_goals, -1)
        obj_diff3 = torch.cat((obj_diff, zero_padding), dim=-1)

        # Nx4, NxGx3 -> (N*G)x4, (N*G)x3
        loc_ori = loc_ori.repeat_interleave(self.sim.n_goals, dim=0, output_size=self.sim.n_all_bots * self.sim.n_goals)
        obj_diff3 = obj_diff3.reshape(-1, 3)

        # (N*G)x4, (N*G)x3 -> (N*G)x3
        loc_diff3 = apply_quat_rot(loc_ori, obj_diff3)

        # (N*G)x3 -> N*G -> NxG
        obj_in_fov = check_fov(loc_diff3).reshape(self.sim.n_all_bots, self.sim.n_goals)
        self.obj_in_frame = obj_in_clear & obj_in_fov  # & (obj_dist < cfg.MAX_RECOG_DIST)

        return obj_diff, obj_dist, obj_in_clear_arr

    # --------------------------------------------------------------------------
    # MARK: get_prox_data

    def get_prox_data(self, obj_diff: Tensor, obj_dist: Tensor, bot_ori: Tensor) -> 'tuple[Tensor, ...]':
        """
        Detect near signal strength (proximity for entities in clipped range)
        wrt. 4 oriented receivers, each covering an angle of 90 degrees.
        """

        # Pairwise distances
        # Nx2 -> ExBx2 -> Ex1xBx2 - ExBx1x2 -> ExBxBx2 -> NxBx2
        bot_pos = self.bot_pos.reshape(self.sim.n_envs, -1, 2)
        bot_diff = bot_pos.unsqueeze(1) - bot_pos.unsqueeze(2)
        bot_diff = bot_diff.reshape(self.sim.n_all_bots, -1, 2)

        # Suppress own signal in distance calculation
        bot_dist = torch.linalg.norm(bot_diff, dim=-1)
        bot_dist[bot_dist == 0.] = torch.inf

        # NxBx0|2, NxGx0|2 -> Nx(B+G)x0|2
        src_dist = torch.cat((bot_dist, obj_dist), dim=1)
        src_diff = torch.cat((bot_diff, obj_diff), dim=1)

        # Proximity weights via truncated inverse prop. characteristic
        sig_strength = norm_distance(src_dist, cfg.SIGNAL_RADIUS).mul_(2.)

        # Angle-range weights
        # NOTE: Own angles are arbitrary, but their weight should already be suppressed
        incoming_angle = torch.atan2(src_diff[..., 1], src_diff[..., 0])
        rcvr_angles = clip_angle_range(get_eulz_from_quat(bot_ori) + self.rcvr_rel_phases)

        # Nx(B+G), Nx4 -> Nx(B+G)x1 - Nx1x4 -> Nx(B+G)x4
        rel_angles = incoming_angle.unsqueeze(-1) - rcvr_angles.unsqueeze(1)
        rel_angles = clip_angle_range(rel_angles)

        # Phase-shifted filters
        rcvr_weights = rel_angles.cos_().clip_(0.)

        # Combined weights aggregated over sig. srcs.
        # Nx(B+G)x4, Nx(BxG) -> Nx(B+G)x4 -> Nx4
        prox_sig_agg = rcvr_weights.mul_(sig_strength.unsqueeze(-1)).sum(1)

        return bot_dist, src_dist, prox_sig_agg

    # --------------------------------------------------------------------------
    # MARK: get_imu_data

    def get_imu_data(self, bot_ori: Tensor, loc_ori: Tensor) -> 'tuple[Tensor, ...]':
        """Simulate the accelerometer, gyroscope, and AHRS."""

        bot_vel = self.env_bot_vel.reshape(-1, 2).contiguous()

        # Simulate accelerometer
        acc = (bot_vel - self.bot_vel) / self.dt
        acc = apply_quat_rot(loc_ori, torch.cat((acc, self.zero_column), dim=-1))[:, :2]

        # Expecting at most |0. - 0.3m/0.25s| / 0.25s at abrupt stop from full speed
        acc /= 5.

        self.bot_vel = bot_vel

        # Expecting at most one turn per 6 seconds
        ang_vel = self.env_bot_ang_vel.reshape(-1, 1)

        # AHRS
        sin_cos_z = get_trigonz_from_quat(bot_ori)

        return acc, ang_vel, sin_cos_z

    # --------------------------------------------------------------------------
    # MARK: eval_bot_tasks

    def eval_bot_tasks(self, bot_pos_arr: np.ndarray, obj_in_clear_arr: Tensor, loc_ori: Tensor) -> 'tuple[Tensor,...]':
        """Check if any goals are in reach and confirm completed tasks."""

        goal_in_clear_arr = obj_in_clear_arr[self.row_idx_arr, self.goal_idx_arr]

        # Get A* path length and current direction
        slices = [slice(i, i+self.sim.n_bots) for i in range(0, self.sim.n_all_bots, self.sim.n_bots)]

        env_path_est = [
            env.get_path_estimates(
                bot_pos_arr[idcs],
                self.goal_pos_arr[idcs],
                goal_in_clear_arr[idcs])
            for env, idcs in zip(self.sim.envs, slices)]

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

        obj_undone_mask_f = self.goal_mask_f * bot_undone_mask_f.unsqueeze(-1)
        obj_masked_path_len = obj_undone_mask_f * goal_path_len.unsqueeze(-1) / cfg.MAX_GOAL_DIST

        return new_bot_done, goal_dir, goal_path_dir, goal_dist, obj_masked_path_len, obj_undone_mask_f

    # --------------------------------------------------------------------------
    # MARK: eval_env_reset

    def eval_env_reset(self) -> Tensor:
        """Step time and check for terminal envs. ahead of next sim./eval. step."""

        self.env_step_ctrs += 1
        env_rst_mask = (self.env_step_ctrs == self.steps_in_ep) | self.bot_done_mask.reshape(self.sim.n_envs, -1).all(1)

        self.env_rst_idcs = torch.nonzero(env_rst_mask).flatten().tolist()
        self.bot_rst_mask = env_rst_mask.repeat_interleave(self.sim.n_bots, output_size=self.sim.n_all_bots)

        env_run_times = self.env_step_ctrs * self.dt
        time_left = (self.ep_duration - env_run_times) / 60.
        time_left = time_left.repeat_interleave(self.sim.n_bots, output_size=self.sim.n_all_bots)

        return time_left

    # --------------------------------------------------------------------------
    # MARK: eval_rewards

    def eval_rewards(self, new_bot_done: Tensor, src_dist: Tensor) -> 'tuple[Tensor, ...]':
        """Check conditions for joint and individual rewards."""

        sim = self.sim

        # Joint rewards
        if self.belief_gain:
            goal_pred_ok = torch.linalg.norm(self.goal_pos - self.goal_pos_in_mind, dim=-1) < cfg.GOAL_PRED_RADIUS
            goal_pred_ok = (goal_pred_ok | self.bot_done_mask).float()

            goal_pred_delta = (goal_pred_ok - self.goal_pred_ok).sign()
            goal_pred_delta = goal_pred_delta.reshape(sim.n_envs, sim.n_bots).sum(1) / sim.n_bots_per_goal

            self.goal_pred_ok = goal_pred_ok

        else:
            goal_pred_delta = 0.

        # Event when a bot correctly recognises that a spec. obj. is in sight
        obj_found = self.obj_in_frame & self.obj_in_mind

        # NxG -> ExG -> E
        obj_found = obj_found.reshape(sim.n_envs, sim.n_bots, -1).any(1)
        new_obj_found = ~self.obj_found & obj_found
        any_obj_found = new_obj_found.any(1)

        if self.time_table is not None:
            if any_obj_found.any():
                goal_found_mask = new_obj_found.unsqueeze(1) * self.goal_mask_f.reshape(sim.n_envs, sim.n_bots, -1)
                goal_found_time = goal_found_mask.sum(-1) * self.env_step_ctrs.unsqueeze(1)
                self.time_table[-1, ..., 0] += goal_found_time.long()

            if new_bot_done.any():
                goal_reached_time = new_bot_done.reshape(sim.n_envs, sim.n_bots) * self.env_step_ctrs.unsqueeze(1)
                self.time_table[-1, ..., 1] += goal_reached_time.long()

        self.obj_found |= obj_found
        goal_found = self.obj_found.repeat_interleave(
            sim.n_bots, dim=0, output_size=sim.n_all_bots)[self.row_idcs, self.goal_idx]

        # Mark envs. to put a short sequence of steps into the aux. task prio. buffer
        prio_event_mask = any_obj_found

        joint_rwd = any_obj_found * cfg.OBJ_FOUND_RWD + goal_pred_delta * cfg.BLIF_DELTA_RWD

        # Long-horizon indiv. rewards
        bot_cell_idcs = torch.bucketize(self.bot_pos, self.side_delims)
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
        near_src_prox = (1. - src_dist / (2.*cfg.BOT_WIDTH)).clip(0.).sum(-1)

        indiv_pen = colliding * cfg.COLLISION_RWD + near_src_prox * cfg.PROXIMITY_RWD

        # Stack joint and individual rewards
        joint_rwd = joint_rwd.repeat_interleave(sim.n_bots, output_size=sim.n_all_bots)
        rwd = torch.stack((joint_rwd, indiv_rwd, indiv_pen), dim=-1)

        return prio_event_mask, rwd, goal_found, colliding, near_src_prox

    # --------------------------------------------------------------------------
    # MARK: async_step_graphics

    async def async_step_graphics(self, other_task: asyncio.Task):
        self.gym.step_graphics(self.sim.handle)

        if self.render_cameras:
            self.gym.render_all_camera_sensors(self.sim.handle)

        if not self.headless:
            self.interface.sync_redraw()

        await other_task
        self.async_event_loop.stop()

    # --------------------------------------------------------------------------
    # MARK: get_image_observations

    def get_image_observations(self) -> Tensor:
        if not self.render_cameras:
            return self.null_obs_img

        self.gym.start_access_image_tensors(self.sim.handle)

        rgb = torch.stack(self.img_rgb_list)[..., :3]
        dep = torch.stack(self.img_dep_list)
        seg = torch.stack(self.img_seg_list)

        # Normalise
        rgb = rgb / 255.
        dep = norm_distance(-dep, cfg.MAX_IMG_DEPTH)

        # Override black sky colour
        sky_mask = (seg == cfg.ENT_CLS_SKY).unsqueeze(-1)
        rgb = torch.where(sky_mask, self.sky_clr, rgb)

        # Convert to HSV space and put channels before spatial dims.
        if self.keep_rgb_over_hsv:
            clr = rgb.permute(0, 3, 1, 2)

        else:
            clr = rgb_to_hsv(rgb, stack_dim=1)

        # Stack channels
        if self.keep_segmentation:
            obs_img = torch.cat((clr, dep.unsqueeze(1), seg.unsqueeze(1)), dim=1)

        else:
            obs_img = torch.cat((clr, dep.unsqueeze(1)), dim=1)

        self.gym.end_access_image_tensors(self.sim.handle)

        return obs_img

    # --------------------------------------------------------------------------
    # MARK: run_async_ops

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
