"""Simulation control flow and runtime"""

import tkinter as tk
import os
from argparse import Namespace

import numpy as np
from cv2 import VideoWriter, VideoWriter_fourcc, imwrite
from isaacgym import gymapi, gymutil
import torch
from torch import Tensor

from discit.optim import AnnealingScheduler, CoeffScheduler, MultiOptimizer, MultiScheduler, NAdamW
from discit.marl import MAXPPO
from discit.track import CheckpointTracker

import config as cfg
from sim import MazeSim
from task import BasicInterface, MazeTask
from train import BeliefAuxTask, VisionAuxTask
from model import ActorCritic, RandomActorCritic, VisNet
from utils import get_available_file_idx
from utils_torch import apply_quat_rot, norm_distance


# ------------------------------------------------------------------------------
# MARK: Interface

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

    OFFSET_CAM_AHEAD = [[cfg.CAM_OFFSET[0], 0., 0.]]
    OFFSET_CAM_ABOVE = [[0., 0., cfg.CAM_OFFSET[2]]]

    OFFSET_3RD_AHEAD = [[-cfg.BOT_WIDTH*2, 0., 0.]]
    OFFSET_3RD_ABOVE = [[0., 0., cfg.BOT_HEIGHT*1.42]]

    OBJ_LINE_NUM = 16
    OBJ_LINE_OFFSET = 0.01
    OBJ_LINE_CLR = (1., 0., 0.)

    DIR_LINE_OFFSET = 0.01
    DIR_LINE_LENGTH = cfg.BOT_RADIUS * 2.
    DIR_LINE_CLR_DEFAULT = (1., 0., 0.)
    DIR_LINE_CLR_GOAL_SEEN = (0., 1., 0.)
    DIR_LINE_CLR_OBJ_SEEN = (0., 0., 1.)
    DIR_LINE_CLR_TASK_DONE = (1., 1., 1.)

    PREV_ZOOM = 9
    PREV_DIM = (cfg.OBS_IMG_RES_WIDTH, 3 * cfg.OBS_IMG_RES_HEIGHT)

    ENT_CLASSES = np.linspace(0, 255, cfg.N_ENT_CLASSES, dtype=np.float32)[:, None, None]
    RGB_CLASSES = np.round(np.array(sum(cfg.COLOURS.values(), start=[]), dtype=np.float32)[:, None, None] * 255.)

    OBS_EVENTS = [
        (gymapi.KEY_ESCAPE, 'end_session'), (gymapi.KEY_L, 'lvl_reset'),
        (gymapi.KEY_H, 'print_help'), (gymapi.KEY_G, 'print_debug'),
        (gymapi.KEY_N, 'cycle_env'), (gymapi.KEY_B, 'cycle_bot'),
        (gymapi.KEY_V, 'cycle_view'), (gymapi.KEY_I, 'save_view')]

    ACT_EVENTS = [
        (gymapi.KEY_W, 'move_forw'), (gymapi.KEY_S, 'move_back'),
        (gymapi.KEY_A, 'move_left'), (gymapi.KEY_D, 'move_right'),
        (gymapi.KEY_SPACE, 'alt_action'), (gymapi.KEY_C, 'change_preview')]

    OBS_KEYS = set(name for _, name in OBS_EVENTS)
    ACT_KEYS = set(name for _, name in ACT_EVENTS)

    MKBD_EVENTS = OBS_EVENTS + ACT_EVENTS
    MKBD_KEYS = OBS_KEYS | ACT_KEYS

    VIEW_CAMERA = 2
    VIEW_FOLLOW = 1
    VIEW_FREE = 0

    # --------------------------------------------------------------------------
    # MARK: init

    def __init__(self, session: 'Session', sim: MazeSim, device: str):
        super().__init__(sim.gym, sim.handle)

        self.session = session
        self.sim = sim

        # Initial view
        self.view = self.VIEW_FREE
        self.env_idx = 0
        self.bot_idx = 0
        self.all_bot_idx = 0
        self.all_actor_idx: int = None

        side_length = sim.envs[0].sampler.side_length
        self.viewer_top_pos = gymapi.Vec3(0., -side_length*0.6, side_length*0.6)
        self.viewer_top_target = gymapi.Vec3(0., -side_length*0.15, 0.)

        self.set_top_view()

        # Setup input events for user interaction
        for key, name in self.MKBD_EVENTS:
            self.gym.subscribe_viewer_keyboard_event(self.viewer, key, name)

        self.offset_cam_ahead = torch.tensor(self.OFFSET_CAM_AHEAD, dtype=torch.float32)
        self.offset_cam_above = torch.tensor(self.OFFSET_CAM_ABOVE, dtype=torch.float32)
        self.offset_3rd_ahead = torch.tensor(self.OFFSET_3RD_AHEAD, dtype=torch.float32)
        self.offset_3rd_above = torch.tensor(self.OFFSET_3RD_ABOVE, dtype=torch.float32)

        self.sky_clr = torch.tensor(self.RGB_CLASSES[cfg.SKY_CLR_IDX, 0, 0], device=device)

        # Debugging
        env = sim.envs[0]
        n_bots, n_goals = env.sampler.n_bots, env.sampler.n_goals

        self.obj_line_num = n_goals * self.OBJ_LINE_NUM
        obj_line_angles = np.arange(self.OBJ_LINE_NUM) * (2.*np.pi/self.OBJ_LINE_NUM)
        obj_line_pts = np.stack((np.cos(obj_line_angles), np.sin(obj_line_angles)), axis=-1) * cfg.GOAL_ZONE_RADIUS
        self.obj_line_start_pts = np.tile(obj_line_pts, (n_goals, 1)).astype(np.float32)
        self.obj_line_end_pts = np.concatenate((self.obj_line_start_pts[1:], self.obj_line_start_pts[:1]))
        self.obj_line_offset = np.array((self.OBJ_LINE_OFFSET,)*self.obj_line_num, dtype=np.float32)[:, None]
        self.obj_line_clr = np.array((self.OBJ_LINE_CLR,)*self.obj_line_num, dtype=np.float32)

        self.dir_line_offset = np.array((self.DIR_LINE_OFFSET,)*n_bots, dtype=np.float32)[:, None]
        self.dir_line_clr_dft = torch.tensor((self.DIR_LINE_CLR_DEFAULT,)*n_bots, dtype=torch.float32, device=device)
        self.dir_line_clr_goal = torch.tensor((self.DIR_LINE_CLR_GOAL_SEEN,)*n_bots, dtype=torch.float32, device=device)
        self.dir_line_clr_obj = torch.tensor((self.DIR_LINE_CLR_OBJ_SEEN,)*n_bots, dtype=torch.float32, device=device)
        self.dir_line_clr_done = torch.tensor((self.DIR_LINE_CLR_TASK_DONE,)*n_bots, dtype=torch.float32, device=device)

        # Manual control
        self.torque_states = {
            key: torch.tensor(val, dtype=torch.float32, device=device)
            for key, val in self.TORQUE_STATES.items()}

        self.key_vec = np.zeros(5)

        # Side panel
        self.prev_reconstruct = False

        if session.preview:
            self.tk_root = tk.Tk()
            self.tk_canvas = tk.Canvas(
                self.tk_root,
                width=self.PREV_DIM[0] * self.PREV_ZOOM,
                height=self.PREV_DIM[1] * self.PREV_ZOOM)

            self.tk_canvas.pack()

            self.visnet = VisNet().to(device)
            self.visnet.load_state_dict(torch.load(os.path.join(cfg.ASSET_DIR, 'visnet.pt'), map_location=device))

        else:
            self.tk_root = self.tk_canvas = self.visnet = None

        # Workaround as `get_viewer_camera_handle` produces a segmentation fault
        self.camera = ViewerCamera(sim)

    # --------------------------------------------------------------------------
    # MARK: cycle_target_indices

    def cycle_target_indices(self, env_inc: int = 0, bot_inc: int = 0):
        self.env_idx = (self.env_idx + env_inc) % self.sim.n_envs
        env = self.sim.envs[self.env_idx]

        self.bot_idx = (self.bot_idx + bot_inc) % env.sampler.n_bots
        self.all_bot_idx = env.bot_idx + self.bot_idx
        self.all_actor_idx = self.session.all_bot_idcs[self.all_bot_idx].item()

    # --------------------------------------------------------------------------
    # MARK: set_top_view

    def set_top_view(self):
        if self.view != self.VIEW_FREE:
            return

        self.gym.viewer_camera_look_at(
            self.viewer,
            self.sim.envs[self.env_idx].handle,
            self.viewer_top_pos,
            self.viewer_top_target)

    # --------------------------------------------------------------------------
    # MARK: update_obj_lines

    def update_obj_lines(self):
        env = self.sim.envs[0]

        obj_pos = np.repeat(self.session.obj_pos_arr[env.data.obj_goal_map, 0], self.OBJ_LINE_NUM, axis=0)

        self.gym.add_lines(
            self.viewer,
            env.handle,
            self.obj_line_num,
            np.concatenate((
                obj_pos + self.obj_line_start_pts, self.obj_line_offset,
                obj_pos + self.obj_line_end_pts, self.obj_line_offset), axis=-1),
            self.obj_line_clr)

    # --------------------------------------------------------------------------
    # MARK: update_dir_lines

    def update_dir_lines(self):
        env = self.sim.envs[0]
        bot_pos = self.session.bot_pos_arr[env.bot_slice]

        # Show the agents' internal dir. estimation
        if self.prev_reconstruct:
            obj_in_frame = self.session.obj_in_mind[env.bot_slice]
            goal_pos = self.session.goal_pos_in_mind[env.bot_slice]

        # Show the true direction
        else:
            obj_in_frame = self.session.obj_in_frame[env.bot_slice]
            goal_pos = self.session.goal_pos[env.bot_slice]

        goal_in_frame = obj_in_frame[self.session.row_idcs[env.bot_slice], self.session.goal_idx[env.bot_slice]].float()
        goal_complete = self.session.bot_done_mask_f[env.bot_slice]

        goal_diff = goal_pos - self.session.bot_pos[env.bot_slice]
        goal_diff = goal_diff.sign() * goal_diff.abs().clip(cfg.MIN_GOAL_DIST)
        goal_dist = torch.linalg.norm(goal_diff, dim=-1, keepdim=True)

        goal_dir = goal_diff / goal_dist * self.DIR_LINE_LENGTH
        goal_prox = 1. - goal_dist / cfg.MAX_GOAL_DIST

        dir_line_clr = torch.where(
            obj_in_frame.any(-1, keepdim=True),
            torch.lerp(self.dir_line_clr_obj, self.dir_line_clr_goal, goal_in_frame.unsqueeze(-1)),
            torch.lerp(self.dir_line_clr_dft, self.dir_line_clr_done, goal_complete.unsqueeze(-1)))

        goal_dir = goal_dir.cpu().numpy()
        dir_line_clr = (dir_line_clr * goal_prox).cpu().numpy()

        self.gym.add_lines(
            self.viewer,
            env.handle,
            env.sampler.n_bots,
            np.concatenate((
                bot_pos, self.dir_line_offset,
                bot_pos + goal_dir, self.dir_line_offset), axis=-1),
            dir_line_clr)

    # --------------------------------------------------------------------------
    # MARK: get_viewer_pos

    def get_viewer_pos(self) -> 'tuple[np.ndarray, np.ndarray] | None':
        if self.view == Interface.VIEW_CAMERA:
            offset_ahead = self.offset_cam_ahead
            offset_above = self.offset_cam_above

        elif self.view == Interface.VIEW_FOLLOW:
            offset_ahead = self.offset_3rd_ahead
            offset_above = self.offset_3rd_above

        else:
            return

        self.gym.fetch_results(self.sim_handle, True)
        self.gym.refresh_actor_root_state_tensor(self.sim_handle)

        bot_state = self.session.actor_states[self.all_actor_idx, :7].cpu()
        bot_pos = bot_state[:3]
        bot_ori = bot_state[None, 3:]

        vec_ahead = apply_quat_rot(bot_ori, offset_ahead)[0]
        ref_ahead = apply_quat_rot(bot_ori, self.offset_cam_ahead)[0]

        pos_view = bot_pos + offset_above[0] + vec_ahead
        pos_target = pos_view + ref_ahead

        pos_view = gymapi.Vec3(*pos_view.numpy())
        pos_target = gymapi.Vec3(*pos_target.numpy())

        return pos_view, pos_target

    # --------------------------------------------------------------------------
    # MARK: update_view

    def update_view(self, update_lines: bool = True):
        if self.view == Interface.VIEW_FREE:
            if update_lines:
                self.gym.clear_lines(self.viewer)
                self.update_obj_lines()
                self.update_dir_lines()

            return

        self.gym.viewer_camera_look_at(self.viewer, self.sim.envs[self.env_idx].handle, *self.get_viewer_pos())

    # --------------------------------------------------------------------------
    # MARK: get_torque_from_key_vec

    def get_torque_from_key_vec(self) -> Tensor:
        """Compute 4-wheel torques from keyboard press state."""

        mvmt_forw = self.key_vec[0:2].sum()
        mvmt_left = self.key_vec[2:4].sum()
        scale = (cfg.MOT_MAX_TORQUE / 2.) if self.key_vec[4] else cfg.MOT_MAX_TORQUE

        if not (mvmt_forw or mvmt_left):
            return self.torque_states[np.nan]

        return self.torque_states.get(np.arctan2(mvmt_left, mvmt_forw), self.torque_states[np.nan]) * scale

    # --------------------------------------------------------------------------
    # MARK: get_debug_info

    def get_debug_info(self) -> str:
        data = self.session.last_data

        if data is None:
            return 'State not yet evaluated.\n'

        obs_img, obs_vec, obs_map = data['obs']

        if self.session.preview:
            seg = obs_img[self.all_bot_idx, -1].long()
            n_obj_px = (seg == cfg.ENT_CLS_OBJ).long().sum().item()

            with torch.inference_mode():
                b_obj = self.visnet(obs_img[:, :cfg.OBS_IMG_CHANNELS])[2]

            b_obj = b_obj[self.all_bot_idx]
            m_obj = b_obj == b_obj.max()
            self.session.obj_in_mind[self.all_bot_idx] = m_obj[1:]

            m_obj = m_obj.cpu().numpy()
            p_obj = b_obj.softmax().mul_(100.).cpu().numpy()

        else:
            n_obj_px = 0
            m_obj = p_obj = [0.]*9

        xidx, yidx = torch.bucketize(self.session.bot_pos[self.all_bot_idx], self.session.max_delims)

        obs_img = obs_img[self.all_bot_idx].cpu().mean((-2, -1)).numpy()
        obs_vec = obs_vec[self.all_bot_idx].cpu().numpy()
        obs_map = obs_map[self.env_idx, :, xidx, yidx].cpu().numpy()

        joint_rwd, indiv_rwd, indiv_pen = data['rwd'][self.all_bot_idx].cpu().numpy()

        aux_val = data['vaux'][self.all_bot_idx].cpu().numpy()
        prio_event_flag = data['prio'][self.env_idx].item()
        nrst_mask_f = data['nrst'][self.all_bot_idx].item()

        score = self.session.get_metrics()['score']
        n_bots_per_goal = self.sim.envs[self.env_idx].sampler.n_bots_per_goal

        env_run_times = self.session.env_step_ctrs * self.session.dt
        time_left = (self.session.ep_durations - env_run_times)[self.env_idx]
        # time_left = (time_left / 60.).repeat_interleave(self.session.all_n_bots, output_size=self.sim.n_all_bots)

        return (
            '\n--------------------------------------------------------------\n'
            'SESSION\n'
            f'Avg. score (%)  | {score: .2f}\n'
            f'Joint reward    | {joint_rwd: .2f}\n'
            f'Indiv. reward   | {indiv_rwd: .2f}\n'
            f'Indiv. penalty  | {indiv_pen: .2f}\n'
            f'Obj. sight tgt. | {"|".join("  X " if flag else " __ " for flag in aux_val[:9])}\n'
            f'Obj. sight b.m. | {"|".join("  X " if flag else " __ " for flag in m_obj)}\n'
            f'Obj. sight b.p. | {"|".join(f" {val:2.0f} " for val in p_obj)}\n'
            f'Obj. sight npx. | {n_obj_px} ({100 * n_obj_px / (cfg.OBS_IMG_RES_WIDTH*cfg.OBS_IMG_RES_HEIGHT):.2f}%)\n'
            f'Goal pos. tgt.  | X: {aux_val[9] * cfg.MAX_GOAL_DIST: .2f} | Y: {aux_val[10] * cfg.MAX_GOAL_DIST: .2f}\n'
            f'Prio. evt. flag | {prio_event_flag}\n'
            f'Reset flag      | {not bool(nrst_mask_f)}\n'
            f'Time left (s)   | {time_left: .2f}\n\n'

            'MAP\n'
            f'Obj. in cell    | {"|".join(" X " if flag else " _ " for flag in obs_map[:8])}\n'
            f'Wall in cell    |        N: {"X" if obs_map[8] else "_"}\n'
            f'                | W: {"X" if obs_map[9] else "_"} |      | E: {"X" if obs_map[11] else "_"}\n'
            f'                |        S: {"X" if obs_map[10] else "_"}\n'
            f'Cell clr. cls.  | {obs_map[12]: .2f}\n\n'

            'OBSERVATION\n'
            f'Avg. img. chan. | R: {obs_img[0]: .2f} | G: {obs_img[1]: .2f} | B: {obs_img[2]: .2f}\n'
            f'                | D: {obs_img[3]: .2f}\n'
            f'Act. torques    | FL: {obs_vec[0]: .2f} | FR: {obs_vec[1]: .2f}\n'
            f'                | BL: {obs_vec[2]: .2f} | BR: {obs_vec[3]: .2f}\n'
            f'Act. light      | {"ON" if obs_vec[4] else "OFF"}\n'
            f'GPS bot pos.    | X: {obs_vec[5] * cfg.MAX_GOAL_DIST: .2f} | Y: {obs_vec[6] * cfg.MAX_GOAL_DIST: .2f}\n'
            f'GPS bot vel.    | X: {obs_vec[7]: .2f} | Y: {obs_vec[8]: .2f}\n'
            f'IMU accel.      | X: {obs_vec[9]: .2f} | Y: {obs_vec[10]: .2f}\n'
            f'IMU ang. vel.   | Z: {obs_vec[11]: .2f}\n'
            f'AHRS bot ori.   | SIN(Z): {obs_vec[12]: .2f} | COS(Z): {obs_vec[13]: .2f}\n'
            f'Prox. channels  |            F: {obs_vec[14]: .2f}\n'
            f'                | L: {obs_vec[15]: .2f} |          | R: {obs_vec[17]: .2f}\n'
            f'                |            B: {obs_vec[16]: .2f}\n\n'

            'TASK\n'
            f'Goal spec.      | {"|".join(" X " if flag else " _ " for flag in obs_vec[18:26])}\n'
            f'Goal reached    | {not any(obs_vec[18:26])}\n'
            f'Orac. pos. tgt. | X: {obs_vec[26]: .2f} | Y: {obs_vec[27]: .2f}\n\n'

            'STAT\n'
            f'Near bot dist.  | {obs_vec[28]: .2f}\n'
            f'Vel. norm       | {obs_vec[29]: .2f}\n'
            f'Tasks left      | {"|".join(f" {round(val * n_bots_per_goal):2d} " for val in obs_vec[30:38])}\n'
            f'Avg. dist. left | {"|".join(f" {val * cfg.MAX_GOAL_DIST:2.0f} " for val in obs_vec[38:46])}\n'
            f'Min. dist. left | {"|".join(f" {val * cfg.MAX_GOAL_DIST:2.0f} " for val in obs_vec[46:54])}\n'
            f'Obj. found      | {"|".join("  X " if flag else " __ " for flag in obs_vec[54:62])}\n\n'

            'OBJECT\n'
            f'Obj. dists.     | {"|".join(f" {val * cfg.MAX_GOAL_DIST:2.0f} " for val in obs_vec[78:86])}\n'
            f'Obj. in frame   | {"|".join("  X " if flag else " __ " for flag in obs_vec[86:94])}\n\n'

            'PROGRESS\n'
            f'Goal pos.       | X: {obs_vec[94] * cfg.MAX_GOAL_DIST: .2f} | Y: {obs_vec[95] * cfg.MAX_GOAL_DIST: .2f}\n'
            f'Air direction   | Front: {obs_vec[96]: .2f} | Left: {obs_vec[97]: .2f}\n'
            f'Air dist.       |        {obs_vec[98] * cfg.MAX_GOAL_DIST: .2f}\n'
            f'A*  direction   | Front: {obs_vec[99]: .2f} | Left: {obs_vec[100]: .2f}\n'
            f'A*  path len.   |        {obs_vec[101] * cfg.MAX_GOAL_DIST: .2f}\n'
            f'Goal found      | {bool(obs_vec[102])}\n\n'

            'REWARD\n'
            f'Goal pred. true | {bool(obs_vec[103])}\n'
            f'Goal path delta | {obs_vec[104]: .2f}\n'
            f'Cell reward sum | {obs_vec[105] * cfg.MAX_GOAL_REACHED_RWD: .2f}\n'
            f'Near bot prox.  | {obs_vec[106]: .2f}\n'
            f'Contact flag    | {bool(obs_vec[107])}\n\n')

    # --------------------------------------------------------------------------
    # MARK: get_rendered_images

    def get_rendered_images(self) -> 'tuple[np.ndarray, np.ndarray, np.ndarray]':
        self.gym.render_all_camera_sensors(self.sim_handle)
        self.gym.start_access_image_tensors(self.sim_handle)

        rgb = self.session.img_rgb_list[self.all_bot_idx][..., :3]
        dep = self.session.img_dep_list[self.all_bot_idx]
        seg = self.session.img_seg_list[self.all_bot_idx]

        # Revert to original segmentation tags
        seg[(seg >= cfg.WALL_CLS_OFFSET) & (seg < cfg.OBJ_CLS_OFFSET)] = cfg.ENT_CLS_WALL
        seg[(seg >= cfg.OBJ_CLS_OFFSET) & (seg < cfg.BOT_CLS_OFFSET)] = cfg.ENT_CLS_OBJ
        seg[seg >= cfg.BOT_CLS_OFFSET] = cfg.ENT_CLS_PAYLOAD

        sky_mask = (seg == cfg.ENT_CLS_SKY).unsqueeze(-1)
        rgb = torch.where(sky_mask, self.sky_clr, rgb.float())

        rgb = rgb.cpu().numpy()
        dep = 255. * norm_distance(-dep, cfg.MAX_IMG_DEPTH).cpu().numpy()
        seg = (255. / (cfg.N_ENT_CLASSES - 1)) * seg.cpu().numpy()

        self.gym.end_access_image_tensors(self.sim_handle)

        return rgb, dep, seg

    # --------------------------------------------------------------------------
    # MARK: get_visnet_images

    def get_visnet_images(self) -> 'tuple[np.ndarray, np.ndarray, np.ndarray]':
        obs_img = self.session.last_data['obs'][0][:, :cfg.OBS_IMG_CHANNELS]

        hsvd = obs_img[self.all_bot_idx:self.all_bot_idx+1]

        with torch.inference_mode():
            out, _, b_obj, _ = self.visnet(hsvd)

        if self.prev_reconstruct and self.session.ctrl_mode == Session.CTRL_MAN:
            self.session.obj_in_mind[self.all_bot_idx] = b_obj[0, 1:] == b_obj[0].max()

        clr_logits, ent_logits, dep = out.split(cfg.DEC_IMG_CHANNEL_SPLIT, dim=1)
        clr_probs = clr_logits[0].softmax(dim=0).cpu().numpy()
        ent_probs = ent_logits[0].softmax(dim=0).cpu().numpy()
        dep = dep[0, 0].cpu().numpy()

        dep = np.clip(dep * 255., 0., 255.)
        ent_seg = (ent_probs * self.ENT_CLASSES).sum(axis=0)
        rgb_seg = (clr_probs[..., None] * self.RGB_CLASSES).sum(axis=0)

        return rgb_seg, dep, ent_seg

    # --------------------------------------------------------------------------
    # MARK: save_camera_images

    def save_camera_images(self, reconstruct: bool = False) -> 'tuple[str, str, str]':
        images = (self.get_visnet_images if reconstruct else self.get_rendered_images)()
        filenames = []

        for img, cam_type in zip(images, ('rgb', 'dep', 'seg')):
            file_idx = get_available_file_idx(cfg.DATA_DIR, f'img_{cam_type}cam')
            filename = os.path.join(cfg.DATA_DIR, f'img_{cam_type}cam_{file_idx:02d}.png')

            imwrite(filename, (np.flip(img, axis=-1) if cam_type == 'rgb' else img).astype(np.uint8))
            filenames.append(filename)

        return tuple(filenames)

    # --------------------------------------------------------------------------
    # MARK: save_viewer_image

    def save_viewer_image(self) -> str:
        file_idx = get_available_file_idx(cfg.DATA_DIR, 'img_viewer')
        filename = os.path.join(cfg.DATA_DIR, f'img_viewer_{file_idx:02d}.png')

        imwrite(filename, self.camera.render(self))
        # self.gym.write_viewer_image_to_file(self.viewer, filename)

        return filename

    # --------------------------------------------------------------------------
    # MARK: eval_events

    def eval_events(self):
        """Check for keyboard events, update actions and view."""

        if self.gym.query_viewer_has_closed(self.viewer):
            raise KeyboardInterrupt

        self.update_preview()

        for event in self.gym.query_viewer_action_events(self.viewer):
            cmd_key, cmd_press = event.action, event.value > 0

            if self.paused:
                if cmd_key == 'end_session':
                    if cmd_press:
                        raise KeyboardInterrupt

                    else:
                        continue

                self.paused = False
                self.gym.clear_lines(self.viewer)

                print('Session resumed.')

            if cmd_key in self.OBS_KEYS:
                if not cmd_press:
                    continue

                if cmd_key == 'end_session':
                    self.paused = True

                    print('Session paused. Press ESC to exit or another key to resume.')

                elif cmd_key == 'lvl_reset':
                    self.session.reset_all()

                    print('All environments flagged for immediate reset.')

                elif cmd_key == 'print_help':
                    print('Available keys:')

                    for key_map in sorted(self.MKBD_EVENTS, key=lambda k: k[1]):
                        print(key_map)

                elif cmd_key == 'print_debug':
                    print(self.get_debug_info())

                # NOTE: Previous bot torques are not automatically reset to zero
                elif cmd_key == 'cycle_env':
                    self.cycle_target_indices(env_inc=-1 if self.key_vec[-1] else 1)

                    if not self.key_vec[-1]:
                        self.set_top_view()

                    print(f'Env/agent index switched to {self.env_idx}/{self.all_bot_idx}.')

                # NOTE: See note above
                elif cmd_key == 'cycle_bot':
                    self.cycle_target_indices(bot_inc=-1 if self.key_vec[-1] else 1)

                    print(f'Bot/agent index switched to {self.bot_idx}/{self.all_bot_idx}.')

                elif cmd_key == 'cycle_view':
                    self.view = (self.view + (-1 if self.key_vec[-1] else 1)) % 3

                    if not self.key_vec[-1]:
                        self.set_top_view()

                    if self.view == self.VIEW_CAMERA:
                        self.gym.clear_lines(self.viewer)

                elif cmd_key == 'save_view':
                    if self.view != self.VIEW_FREE and self.session.render_cameras:
                        for reconstruct in (False, True):
                            filename_rgb, filename_dep, filename_seg = self.save_camera_images(reconstruct)

                            print(
                                f'Saved camera RGB image to: {filename_rgb}\n'
                                f'Saved camera DEP image to: {filename_dep}\n'
                                f'Saved camera SEG image to: {filename_seg}')

                    print(f'Saved viewer image to: {self.save_viewer_image()}')

            elif cmd_key in self.ACT_KEYS:
                if cmd_key == 'change_preview':
                    if not cmd_press:
                        self.prev_reconstruct = not self.prev_reconstruct

                        print(f'Side preview reconstruction is {"ON" if self.prev_reconstruct else "OFF"}.')

                elif self.session.ctrl_mode != Session.CTRL_MAN:
                    if cmd_press:
                        print('Cannot command agents without manual control mode.')

                    continue

                elif cmd_key == 'alt_action':
                    self.key_vec[-1] = event.value
                    self.session.actions[self.all_bot_idx, -1] = event.value

                    if cmd_press:
                        self.session.obj_in_mind[self.all_bot_idx] = self.session.obj_in_frame[self.all_bot_idx]
                        self.session.goal_pos_in_mind[self.all_bot_idx] = self.session.bot_pos[self.all_bot_idx]

                    continue

                elif cmd_key == 'move_forw':
                    self.key_vec[0] = event.value

                elif cmd_key == 'move_back':
                    self.key_vec[1] = -event.value

                elif cmd_key == 'move_left':
                    self.key_vec[2] = event.value

                elif cmd_key == 'move_right':
                    self.key_vec[3] = -event.value

                self.session.actions[self.all_bot_idx, :cfg.N_DOF_MOT] = self.get_torque_from_key_vec()

    # --------------------------------------------------------------------------
    # MARK: update_preview

    def update_preview(self):
        if not self.session.preview or self.session.last_data is None:
            return

        rgb, dep, seg = (self.get_visnet_images if self.prev_reconstruct else self.get_rendered_images)()

        stacked = np.concatenate((
            rgb,
            np.broadcast_to(dep[..., None], rgb.shape),
            np.broadcast_to(seg[..., None], rgb.shape)), axis=0)

        data = f'P6 {self.PREV_DIM[0]} {self.PREV_DIM[1]} 255 '.encode() + stacked.astype(np.uint8).tobytes()

        photoimage = tk.PhotoImage(
            width=self.PREV_DIM[0],
            height=self.PREV_DIM[1],
            data=data,
            format='PPM'
        ).zoom(self.PREV_ZOOM, self.PREV_ZOOM)

        self.tk_canvas.create_image(0, 0, anchor='nw', image=photoimage)
        self.tk_root.update()

    # --------------------------------------------------------------------------
    # MARK: sync_redraw

    def sync_redraw(self, after_eval: bool = True):
        """Draw the scene in the viewer, syncing sim with real-time."""

        if self.session.ctrl_mode == Session.CTRL_AI or self.prev_reconstruct:
            update_lines = self.paused

        else:
            update_lines = after_eval

        self.update_view(update_lines=update_lines)
        self.gym.draw_viewer(self.viewer, self.sim_handle, False)
        self.gym.sync_frame_time(self.sim_handle)

    # --------------------------------------------------------------------------
    # MARK: reset & cleanup

    def reset(self):
        self.key_vec.fill(0)
        self.update_view(update_lines=False)
        self.gym.clear_lines(self.viewer)

    def cleanup(self):
        if self.tk_root is not None:
            self.tk_root.destroy()

        self.gym.destroy_viewer(self.viewer)

        self.camera.cleanup()


# ------------------------------------------------------------------------------
# MARK: ViewerCamera

class ViewerCamera:
    VIDEO_WINDOW_SIZE = (2560, 1440)
    QUAT_TRF = gymapi.Quat(-0.5, -0.5, -0.5, 0.5)

    def __init__(self, sim: MazeSim):
        self.gym = sim.gym
        self.sim_handle = sim.handle
        self.env_handle = sim.envs[0].handle

        self.sky_clr = (np.array(cfg.COLOURS['neutral'][cfg.SKY_CLR_IDX]) * 255.).round()

        camera_props = gymapi.CameraProperties()
        camera_props.width, camera_props.height = self.VIDEO_WINDOW_SIZE
        self.cam_handle = self.gym.create_camera_sensor(self.env_handle, camera_props)

    def render(self, synced_interface: BasicInterface = None) -> np.ndarray:
        if synced_interface is not None:
            trf = self.gym.get_viewer_camera_transform(synced_interface.viewer, self.env_handle)
            trf.r = trf.r * self.QUAT_TRF
            self.gym.set_camera_transform(self.cam_handle, self.env_handle, trf)

        self.gym.step_graphics(self.sim_handle)
        self.gym.render_all_camera_sensors(self.sim_handle)

        rgb = self.gym.get_camera_image(self.sim_handle, self.env_handle, self.cam_handle, gymapi.IMAGE_COLOR)
        seg = self.gym.get_camera_image(self.sim_handle, self.env_handle, self.cam_handle, gymapi.IMAGE_SEGMENTATION)

        sky_mask = (seg == cfg.ENT_CLS_SKY)[..., None]
        rgb = rgb.reshape(*self.VIDEO_WINDOW_SIZE[::-1], 4)[..., :3]
        rgb = np.where(sky_mask, self.sky_clr, rgb)

        return np.flip(rgb, axis=-1).astype(np.uint8)

    def cleanup(self):
        self.gym.destroy_camera_sensor(self.sim_handle, self.env_handle, self.cam_handle)


# ------------------------------------------------------------------------------
# MARK: VideoRecorder

class VideoRecorder(BasicInterface, ViewerCamera):
    def __init__(self, session: 'Session', sim: MazeSim, draw_freq: int):
        ViewerCamera.__init__(self, sim)

        self.session = session
        self.env_idx = 0
        self.bot_idx = 0
        self.all_bot_idx = self.bot_idx
        self.all_actor_idx = sim.envs[0].bot_idcs[self.bot_idx]
        self.viewer = None
        self.paused = False
        self.skip = True
        self.view = Interface.VIEW_FREE

        self.offset_cam_ahead = torch.tensor(Interface.OFFSET_CAM_AHEAD, dtype=torch.float32)
        self.offset_cam_above = torch.tensor(Interface.OFFSET_CAM_ABOVE, dtype=torch.float32)
        self.offset_3rd_ahead = torch.tensor(Interface.OFFSET_3RD_AHEAD, dtype=torch.float32)
        self.offset_3rd_above = torch.tensor(Interface.OFFSET_3RD_ABOVE, dtype=torch.float32)

        side_length = sim.envs[0].sampler.side_length
        viewer_top_pos = gymapi.Vec3(0., -side_length*0.6, side_length*0.6)
        viewer_top_target = gymapi.Vec3(0., -side_length*0.15, 0.)

        self.gym.set_camera_location(self.cam_handle, self.env_handle, viewer_top_pos, viewer_top_target)

        file_idx = get_available_file_idx(cfg.DATA_DIR, 'vid')

        self.video_out = VideoWriter(
            os.path.join(cfg.DATA_DIR, f'vid_{file_idx:02d}.mp4'),
            VideoWriter_fourcc(*'avc1'),
            draw_freq,
            self.VIDEO_WINDOW_SIZE)

    def eval_events(self):
        pass

    def sync_redraw(self, after_eval: bool = True):
        if self.skip:
            self.skip = False
            return

        if (pos := Interface.get_viewer_pos(self)) is not None:
            self.gym.set_camera_location(self.cam_handle, self.env_handle, *pos)

        self.video_out.write(ViewerCamera.render(self))

    def cleanup(self):
        self.video_out.release()
        ViewerCamera.cleanup(self)


# ------------------------------------------------------------------------------
# MARK: Session

class Session(MazeTask):
    """
    The main process descriptor connecting environment, operational,
    and training constructs, and providing several interfacing options.
    """

    CTRL_AI = 3
    CTRL_RL = 2
    CTRL_GEN = 1
    CTRL_MAN = 0

    REC_IMG = 4
    REC_VEC = 3
    REC_PERF = 2
    REC_VIDEO = 1
    REC_NONE = 0

    ARGS = [
        {'name': '--env_cfg', 'type': str, 'default': '1x5', 'help': 'Number/level config. of parallel environments.'},
        {'name': '--clutter_fn', 'type': str, 'default': 'equal', 'help': 'Fn. to set num. of decoy objs. wrt. goals.'},
        {'name': '--global_spawn_prob', 'type': float, 'default': 0., 'help': 'Option to spawn bots across the maze.'},
        {'name': '--ep_duration', 'type': int, 'default': -1, 'help': 'Episode duration override in seconds.'},
        {'name': '--end_step', 'type': int, 'default': -1, 'help': 'Max steps until auto-termination.'},
        {'name': '--ctrl_mode', 'type': int, 'default': CTRL_MAN, 'help': 'Sim/agent control mode.'},
        {'name': '--rec_mode', 'type': int, 'default': REC_NONE, 'help': 'Data category to record.'},
        {'name': '--preview', 'type': int, 'default': 0, 'help': 'Option to view input or recons. images in side GUI.'},
        {'name': '--headless', 'type': int, 'default': 0, 'help': 'Option to run without a viewer.'},
        {'name': '--draw_freq', 'type': int, 'default': 60, 'help': 'Viewer frames per second.'},
        {'name': '--act_freq', 'type': int, 'default': cfg.STEPS_PER_SECOND, 'help': 'Inference steps per second.'},
        {'name': '--transfer_name', 'type': str, 'default': '', 'help': 'Starting model name/ID string.'},
        {'name': '--transfer_ver', 'type': int, 'default': -1, 'help': 'Starting model ckpt. version.'},
        {'name': '--model_name', 'type': str, 'default': 'diablg3', 'help': 'Model name/ID string.'},
        {'name': '--com_mode', 'type': int, 'default': cfg.COM_TARGET, 'help': 'Mode of inter-agent communication.'},
        {'name': '--aux_mode', 'type': int, 'default': cfg.AUX_REPLAY, 'help': 'Mode of auxiliary com. training.'},
        {'name': '--rwd_mode', 'type': int, 'default': cfg.RWD_ALL, 'help': 'Mode of reward composition.'},
        {'name': '--max_com_range', 'type': float, 'default': 25.5, 'help': 'Max. com. interaction distance.'},
        {'name': '--msg_noise_scale', 'type': float, 'default': 0.1, 'help': 'Scale of noise added to a message.'},
        {'name': '--prob_msg_drop', 'type': float, 'default': 0.1, 'help': 'Prob. of a message packet to be dropped.'},
        {'name': '--prob_com_off', 'type': float, 'default': 0.1, 'help': "Prob. of a bot's broadcast to be disabled."},
        {'name': '--prob_actor', 'type': int, 'default': 1, 'help': 'Option to keep probabilistic inference.'},
        {'name': '--rng_seed', 'type': int, 'default': 42, 'help': 'Seed for numpy and torch RNGs.'}]

    def __init__(self, args: Namespace):
        self.end_step: int = args.end_step
        self.ctrl_mode: int = args.ctrl_mode
        self.rec_mode: int = args.rec_mode
        self.rec_data_queue: 'list[Tensor]' = []
        self.preview = bool(args.preview)

        # Resume model state
        self.aux_mode = args.aux_mode
        self.rwd_mode = args.rwd_mode
        self.prob_actor = bool(args.prob_actor)

        self.ckpter = CheckpointTracker(
            args.model_name, cfg.DATA_DIR, args.sim_device, args.rng_seed,
            transfer_name=args.transfer_name,
            ver_to_transfer=args.transfer_ver if args.transfer_ver >= 0 else None,
            reset_step_on_transfer=True)

        self.metastr = str(args)

        if self.ctrl_mode in (self.CTRL_GEN, self.CTRL_RL):
            self.ckpter.logger.info(f'Training with args.:\n{{{self.metastr[10:-1]}}}')

        # Init. Isaac Gym, envs., and state tensors
        if args.headless and self.rec_mode != self.REC_VIDEO:
            args.draw_freq = args.act_freq

        sim = MazeSim(args, self.ckpter.rng)
        self.last_data = None

        if not args.headless:
            interface = Interface(self, sim, args.sim_device)

        elif self.rec_mode == self.REC_VIDEO:
            interface = VideoRecorder(self, sim, args.draw_freq)

        else:
            interface = None

        super().__init__(
            sim,
            interface,
            None if args.ep_duration < 0 else args.ep_duration,
            args.act_freq,
            args.draw_freq,
            render_cameras=self.ctrl_mode > self.CTRL_MAN or self.rec_mode > self.REC_PERF or self.preview,
            keep_segmentation=self.ctrl_mode == self.CTRL_GEN or self.rec_mode > self.REC_PERF or self.preview,
            keep_rgb_over_hsv=self.ctrl_mode == self.CTRL_MAN and not self.preview,
            stagger_env_resets=self.ctrl_mode in (self.CTRL_GEN, self.CTRL_RL),
            reward_belief_gain=self.rwd_mode in (cfg.RWD_GAIN, cfg.RWD_ALL),
            reward_belief_util=self.rwd_mode in (cfg.RWD_UTIL, cfg.RWD_ALL),
            oracular_input=args.com_mode == cfg.COM_ORACULAR,
            track_performance=self.rec_mode == self.REC_PERF,
            device=args.sim_device)

        self.model_options = {
            'n_bots_list': self.all_n_bots.tolist(),
            'com_mode': min(args.com_mode, cfg.COM_TARGET),
            'com_range': args.max_com_range,
            'msg_noise': args.msg_noise_scale,
            'msg_dropout': args.prob_msg_drop}

        if not args.headless:
            interface.cycle_target_indices()

    # --------------------------------------------------------------------------
    # MARK: post_step

    def post_step(
        self,
        obs: 'tuple[Tensor, ...]',
        step_data: 'dict[str, Tensor | tuple[Tensor, ...]]' = None
    ) -> 'tuple[Tensor, ...]':

        if step_data is not None:
            step_data['obs'] = obs

            # Keep data for debugging via interface
            self.last_data = step_data

            # Keep data to save
            if self.rec_mode:
                self.update_rec_data_queue(step_data)

        # Remove segmentation channel
        if self.keep_segmentation:
            obs = (obs[0][:, :cfg.OBS_IMG_CHANNELS], *obs[1:])

        return obs

    # --------------------------------------------------------------------------
    # MARK: update_rec_data_queue

    def update_rec_data_queue(self, step_data: 'dict[str, Tensor | tuple[Tensor, ...]]'):
        img, vec, spa = step_data['obs']
        rwd = step_data['rwd']
        vaux = step_data['vaux']
        prio = step_data['prio']
        nrst = step_data['nrst']

        if self.rec_mode == self.REC_PERF:
            vec_data = torch.hstack((self.goal_pos, self.goal_pos_in_mind)).cpu().numpy()
            self.rec_data_queue.append(vec_data)

            return

        # Images are stored in full or vector form (as means)
        if self.rec_mode == self.REC_VEC:
            img = img.mean((-2, -1))
            spa = spa.mean((-2, -1))

        prio = prio.repeat_interleave(self.all_n_bots, dim=0, output_size=self.sim.n_all_bots).unsqueeze(-1)

        vecs = (vec, rwd, vaux, prio, nrst, self.goal_pos_in_mind)

        img_data = img.cpu().numpy()
        vec_data = torch.hstack(vecs).cpu().numpy()
        map_data = spa.cpu().numpy()

        self.rec_data_queue.extend((img_data, vec_data, map_data))

    # --------------------------------------------------------------------------
    # MARK: save_rec_data

    def save_rec_data(self):
        if self.rec_mode < self.REC_PERF or not self.rec_data_queue:
            return

        file_idx = get_available_file_idx(cfg.DATA_DIR, 'rec')
        filename = os.path.join(cfg.DATA_DIR, f'rec_{file_idx:02d}.npz')

        if self.rec_mode == self.REC_PERF:
            tab = self.time_table.cpu().numpy()
            vec = np.stack(self.rec_data_queue)
            self.rec_data_queue.clear()

            np.savez_compressed(filename, tab=tab, vec=vec, meta=self.metastr)

        else:
            img = np.stack(self.rec_data_queue[0::3])
            vec = np.stack(self.rec_data_queue[1::3])
            spa = np.stack(self.rec_data_queue[2::3])
            self.rec_data_queue.clear()

            np.savez_compressed(filename, img=img, vec=vec, map=spa)

        print(f'Saved data to: {filename}')

    # --------------------------------------------------------------------------
    # MARK: run

    def run(self):
        print(f'Data logging is {"ON" if self.rec_mode else "OFF"}.')

        try:
            if self.ctrl_mode == self.CTRL_RL:
                self.train()

            elif self.ctrl_mode == self.CTRL_GEN:
                self.train_vis()

            elif self.ctrl_mode == self.CTRL_AI:
                self.eval()

            else:
                self.test()

            print('\nEnding...')

        except KeyboardInterrupt:
            print('\nInterrupted...')

        self.save_rec_data()

        # Cleanup
        if self.interface is not None:
            self.interface.cleanup()

        self.sim.cleanup()
        self.async_event_loop.close()

        print('Done.')

    # --------------------------------------------------------------------------
    # MARK: train

    def train(self):

        # Init. model
        model = ActorCritic(**self.model_options)
        model.to(self.ckpter.device)

        # Init. optimizers
        policy_optimizer = NAdamW(
            (param for param in model.policy.parameters() if param.requires_grad),
            lr=cfg.UPDATE_MAP['policy']['lr'],
            weight_decay=cfg.WEIGHT_DECAY,
            device=self.ckpter.device)

        critic_optimizer = NAdamW(
            (param for param in model.valuator.parameters() if param.requires_grad),
            lr=cfg.UPDATE_MAP['critic']['lr'],
            weight_decay=cfg.WEIGHT_DECAY,
            device=self.ckpter.device)

        optimizer = MultiOptimizer(policy=policy_optimizer, critic=critic_optimizer)

        # Load state
        encoder_path = os.path.join(cfg.ASSET_DIR, 'visenc.pt')
        decoder_path = os.path.join(cfg.ASSET_DIR, 'vismlp.pt')

        model.visencoder.load_state_dict(torch.load(encoder_path, map_location=self.ckpter.device))
        model.visdecoder.load_state_dict(torch.load(decoder_path, map_location=self.ckpter.device))

        self.ckpter.restore(model, optimizer)

        # Init. schedulers
        policy_scheduler = AnnealingScheduler(
            policy_optimizer,
            step_milestones=cfg.UPDATE_MILESTONE_MAP['policy'],
            starting_step=self.ckpter.meta['update_step'])

        critic_scheduler = AnnealingScheduler(
            critic_optimizer,
            step_milestones=cfg.UPDATE_MILESTONE_MAP['critic'],
            lr_div_factors=cfg.UPDATE_MAP['critic']['div'],
            starting_step=self.ckpter.meta['update_step'])

        entropy_scheduler = CoeffScheduler(
            cfg.UPDATE_MILESTONE_MAP['policy'][-1],
            cfg.ENT_WEIGHT_MILESTONES,
            starting_step=self.ckpter.meta['update_step'],
            device=self.ckpter.device)

        scheduler = MultiScheduler(policy=policy_scheduler, critic=critic_scheduler, entropy=entropy_scheduler)

        env_bot_idcs = torch.from_numpy(self.sim.env_bot_idcs).to(self.ckpter.device)

        # Init. aux. task
        aux_task = None if self.aux_mode == cfg.AUX_NONE else BeliefAuxTask(
            model.policy,
            policy_optimizer,
            self.ckpter.rng,
            self.sim.n_envs,
            env_bot_idcs,
            cfg.COM_BUFFER_SIZE,
            cfg.N_TRUNCATED_STEPS,
            self.aux_mode == cfg.AUX_ONLINE,
            self.aux_mode == cfg.AUX_DETACH)

        rl_algo = MAXPPO(
            self.step,
            self.ckpter,
            scheduler,
            env_bot_idcs,
            self.sim.n_all_bots,
            cfg.UPDATE_MILESTONE_MAP['policy'][-1],
            cfg.LOG_EPOCH_INTERVAL,
            cfg.CKPT_EPOCH_INTERVAL,
            cfg.BRANCH_EPOCH_INTERVAL,
            cfg.N_ROLLOUT_STEPS,
            cfg.N_TRUNCATED_STEPS,
            cfg.N_PASSES_PER_STEP,
            cfg.BUFFER_SIZE,
            self.sim.n_all_bots,
            cfg.DISCOUNTS,
            cfg.TRACE_LAMBDA,
            cfg.CLIP_RATIO,
            value_weight=cfg.VALUE_WEIGHT,
            aux_weight=cfg.AUX_WEIGHT,
            entropy_weight=entropy_scheduler.value,
            aux_task=aux_task,
            log_dir=cfg.LOG_DIR,
            accelerate=True)

        try:
            rl_algo.run()

        except KeyboardInterrupt:
            rl_algo.writer.close()
            raise

    # --------------------------------------------------------------------------
    # MARK: train_vis

    def train_vis(self):

        # Load model, optimizer, scheduler
        model = RandomActorCritic(**self.model_options)
        model.to(self.ckpter.device)

        optimizer = NAdamW(
            model.visnet.parameters(),
            lr=cfg.UPDATE_MAP['visenc']['lr'],
            weight_decay=cfg.WEIGHT_DECAY,
            device=self.ckpter.device)

        self.ckpter.restore(model, optimizer)

        scheduler = AnnealingScheduler(
            optimizer,
            step_milestones=cfg.UPDATE_MILESTONE_MAP['visenc'],
            starting_step=self.ckpter.meta['update_step'])

        # Init. aux. task
        aux_task = VisionAuxTask(model.visnet, optimizer)

        rl_algo = MAXPPO(
            self.step,
            self.ckpter,
            scheduler,
            self.sim.n_all_bots,
            self.sim.n_all_bots,
            cfg.UPDATE_MILESTONE_MAP['visenc'][-1],
            cfg.VIS_LOG_INTERVAL,
            cfg.VIS_CKPT_INTERVAL,
            cfg.VIS_BRANCH_INTERVAL,
            batch_size=cfg.VIS_BATCH_SIZE,
            policy_weight=0.,
            aux_task=aux_task,
            log_dir=cfg.LOG_DIR)

        try:
            rl_algo.run()

        except KeyboardInterrupt:
            rl_algo.writer.close()
            raise

    # --------------------------------------------------------------------------
    # MARK: eval

    def eval(self):
        model = ActorCritic(**self.model_options)

        # Ignore params. based on training cfg. when loading model params.
        self.ckpter.model = model.to(self.ckpter.device)

        if path_exists := os.path.exists(path := self.ckpter.meta['model_path']):
            state = torch.load(path, map_location=self.ckpter.device)
            del state['n_bots_tensor']
            del state['valuator.n_bots_tensor']
            del state['valuator.env_bot_idcs']
            model.load_state_dict(state, strict=False)

        log_text = f'{"Loaded" if path_exists else "Initialised"} model ver. {self.ckpter.meta["ckpt_ver"]}.'
        self.ckpter.logger.info(log_text)
        print(log_text)

        mem = model.init_mem()

        with torch.inference_mode():
            obs = self.post_step(self.step(get_info=False)[0])
            step_ctr = 0

            while step_ctr != self.end_step:
                actions, beliefs, mem = model.act(obs, mem, sample=self.prob_actor)

                obs, step_data, _ = self.step(actions, beliefs, get_info=False)
                obs = self.post_step(obs, step_data)

                if not step_data['nrst'].all():
                    mem = model.reset_mem(mem, step_data['nrst'])

                step_ctr += 1

                if self.end_step > 0:
                    print(f'\rStep {step_ctr} of {self.end_step} | {100*step_ctr/self.end_step:.2f}%', end='')

    # --------------------------------------------------------------------------
    # MARK: test

    def test(self):
        with torch.inference_mode():
            step_ctr = -1

            while (step_ctr := step_ctr + 1) != self.end_step:
                self.post_step(*self.step(self.actions.clone(), get_info=False)[:2])


# --------------------------------------------------------------------------
# MARK: main

if __name__ == '__main__':
    args = gymutil.parse_arguments(description='Run MazeBots2 session.', custom_parameters=Session.ARGS)
    session = Session(args)
    session.run()
