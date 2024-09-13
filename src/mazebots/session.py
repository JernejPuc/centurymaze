"""Simulation control flow and runtime"""

import tkinter as tk
import os
from argparse import Namespace

import numpy as np
from PIL import Image
from isaacgym import gymapi, gymutil
import torch
from torch import Tensor, float32

from discit.optim import AnnealingScheduler, CoeffScheduler, NAdamW
from discit.rl import PPG
from discit.track import CheckpointTracker

import config as cfg
from sim import MazeSim
from task import BasicInterface, MazeTask
from model import ActorCritic, VisNet
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

    OFFSET_BOT_AHEAD = [[cfg.CAM_OFFSET[0], 0., 0.]]
    OFFSET_BOT_ABOVE = [[0., 0., cfg.CAM_OFFSET[2]]]

    OFFSET_3RD_AHEAD = [[-cfg.BOT_WIDTH*2, 0., 0.]]
    OFFSET_3RD_ABOVE = [[0., 0., cfg.WALL_HEIGHT/2]]

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
    RGB_CLASSES[cfg.FLOOR_CLR_IDX] = 168.  # Override 127, as the ground appears brighter in renders

    OBS_EVENTS = [
        (gymapi.KEY_ESCAPE, 'end_session'), (gymapi.KEY_L, 'lvl_reset'),
        (gymapi.KEY_H, 'print_help'), (gymapi.KEY_G, 'print_debug'),
        (gymapi.KEY_N, 'cycle_env'), (gymapi.KEY_B, 'cycle_bot'),
        (gymapi.KEY_V, 'cycle_view'), (gymapi.KEY_I, 'save_view')]

    ACT_EVENTS = [
        (gymapi.KEY_W, 'move_forw'), (gymapi.KEY_S, 'move_back'),
        (gymapi.KEY_A, 'move_left'), (gymapi.KEY_D, 'move_right'),
        (gymapi.KEY_SPACE, 'light_beacon'), (gymapi.KEY_C, 'change_preview')]

    OBS_KEYS = set(name for _, name in OBS_EVENTS)
    ACT_KEYS = set(name for _, name in ACT_EVENTS)

    MKBD_EVENTS = OBS_EVENTS + ACT_EVENTS
    MKBD_KEYS = OBS_KEYS | ACT_KEYS

    VIEW_BOT = 2
    VIEW_TOP = 1
    VIEW_3RD = 0

    # --------------------------------------------------------------------------
    # MARK: init

    def __init__(self, session: 'Session', sim: MazeSim, device: str):
        super().__init__(sim.gym, sim.handle)

        self.session = session
        self.sim = sim

        # Init viewer
        side_length = sim.data['spec'].item()['grid']['side_length']
        self.viewer_top_pos = gymapi.Vec3(side_length*0.6, 0., side_length*0.6)
        self.viewer_top_target = gymapi.Vec3(side_length*0.15, 0., 0.)

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

        self.obj_line_num = sim.n_goals * self.OBJ_LINE_NUM
        obj_line_angles = np.array([i*2.*np.pi/self.OBJ_LINE_NUM for i in range(self.OBJ_LINE_NUM)], dtype=np.float32)
        obj_line_pts = np.stack((np.cos(obj_line_angles), np.sin(obj_line_angles)), axis=-1) * cfg.GOAL_ZONE_RADIUS
        self.obj_line_start_pts = np.tile(obj_line_pts, (sim.n_goals, 1))
        self.obj_line_end_pts = np.concatenate((self.obj_line_start_pts[1:], self.obj_line_start_pts[:1]))
        self.obj_line_offset = np.array((self.OBJ_LINE_OFFSET,)*self.obj_line_num, dtype=np.float32)[:, None]
        self.obj_line_clr = np.array((self.OBJ_LINE_CLR,)*self.obj_line_num, dtype=np.float32)

        self.dir_line_offset = np.array((self.DIR_LINE_OFFSET,)*sim.n_bots, dtype=np.float32)[:, None]
        self.dir_line_clr_dft = torch.tensor((self.DIR_LINE_CLR_DEFAULT,)*sim.n_bots, dtype=float32, device=device)
        self.dir_line_clr_goal = torch.tensor((self.DIR_LINE_CLR_GOAL_SEEN,)*sim.n_bots, dtype=float32, device=device)
        self.dir_line_clr_obj = torch.tensor((self.DIR_LINE_CLR_OBJ_SEEN,)*sim.n_bots, dtype=float32, device=device)
        self.dir_line_clr_done = torch.tensor((self.DIR_LINE_CLR_TASK_DONE,)*sim.n_bots, dtype=float32, device=device)

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

    # --------------------------------------------------------------------------
    # MARK: cycle_target_indices

    def cycle_target_indices(self, env_inc: int = 0, bot_inc: int = 0):
        self.env_idx = (self.env_idx + env_inc) % self.sim.n_envs
        self.bot_idx = (self.bot_idx + bot_inc) % self.sim.n_bots
        self.all_bot_idx = self.env_idx * self.sim.n_bots + self.bot_idx

    # --------------------------------------------------------------------------
    # MARK: set_top_view

    def set_top_view(self):
        if self.view != self.VIEW_TOP:
            return

        self.gym.viewer_camera_look_at(
            self.viewer,
            self.sim.envs[self.env_idx].handle,
            self.viewer_top_pos,
            self.viewer_top_target)

    # --------------------------------------------------------------------------
    # MARK: update_obj_lines

    def update_obj_lines(self):
        obj_pos = np.repeat(self.session.obj_pos_arr[:, self.all_bot_idx], self.OBJ_LINE_NUM, axis=0)

        self.gym.add_lines(
            self.viewer,
            self.sim.envs[self.env_idx].handle,
            self.obj_line_num,
            np.concatenate((
                obj_pos + self.obj_line_start_pts, self.obj_line_offset,
                obj_pos + self.obj_line_end_pts, self.obj_line_offset), axis=-1),
            self.obj_line_clr)

    # --------------------------------------------------------------------------
    # MARK: update_dir_lines

    def update_dir_lines(self):
        env_slice = slice(self.env_idx * self.sim.n_bots, (self.env_idx+1) * self.sim.n_bots)
        bot_pos = self.session.bot_pos_arr[env_slice]

        # Show the agents' internal dir. estimation
        if self.prev_reconstruct:
            obj_in_frame = self.session.obj_in_mind[env_slice]
            goal_pos = self.session.goal_pos_in_mind[env_slice]

        # Show the true direction
        else:
            obj_in_frame = self.session.obj_in_frame[env_slice]
            goal_pos = self.session.goal_pos[env_slice]

        goal_in_frame = obj_in_frame[self.session.row_idcs[env_slice], self.session.goal_idx[env_slice]].float()
        goal_complete = self.session.bot_done_mask_f[env_slice]

        goal_diff = goal_pos - self.session.bot_pos[env_slice]
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
            self.sim.envs[self.env_idx].handle,
            self.sim.n_bots,
            np.concatenate((
                bot_pos, self.dir_line_offset,
                bot_pos + goal_dir, self.dir_line_offset), axis=-1),
            dir_line_clr)

    # --------------------------------------------------------------------------
    # MARK: update_view

    def update_view(self, update_lines: bool = True):
        if self.view == self.VIEW_TOP:
            if update_lines:
                self.gym.clear_lines(self.viewer)
                self.update_obj_lines()
                self.update_dir_lines()

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
        xidx, yidx = torch.bucketize(self.session.bot_pos[self.all_bot_idx], self.session.side_delims)

        obs_img = obs_img[self.all_bot_idx].cpu().mean((-2, -1)).numpy()
        obs_vec = obs_vec[self.all_bot_idx].cpu().numpy()
        obs_map = obs_map[self.env_idx, :, xidx, yidx].cpu().numpy()

        # TODO: Separate joint and individual reward
        # joint_rwd = data['rwd'][0][self.env_idx].item()
        joint_rwd, indiv_rwd, indiv_pen = data['rwd'][self.all_bot_idx].cpu().numpy()

        aux_val = data['vaux'][self.all_bot_idx].cpu().numpy()
        prio_event_flag = data['prio'][self.env_idx].item()
        nrst_mask_f = data['nrst'][self.all_bot_idx].item()

        score = self.session.get_score()

        return (
            '\n--------------------------------------------------------------\n'
            'SESSION\n'
            f'Avg. score (%)  | {score: .2f}\n'
            f'Joint reward    | {joint_rwd: .2f}\n'
            f'Indiv. reward   | {indiv_rwd: .2f}\n'
            f'Indiv. penalty  | {indiv_pen: .2f}\n'
            f'Obj. sight tgt. | {"|".join(" X " if flag else " _ " for flag in aux_val[:9])}\n'
            f'Goal pos. tgt.  | X: {aux_val[9] * cfg.MAX_GOAL_DIST: .2f} | Y: {aux_val[10] * cfg.MAX_GOAL_DIST: .2f}\n'
            f'Prio. evt. flag | {prio_event_flag}\n'
            f'Reset flag      | {not bool(nrst_mask_f)}\n\n'

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
            f'Goal reached    | {bool(obs_vec[26])}\n'
            f'Time left (s)   | {obs_vec[27] * 60.: .2f}\n\n'

            'STAT\n'
            f'Near bot dist.  | {obs_vec[28]: .2f}\n'
            f'Vel. norm       | {obs_vec[29]: .2f}\n'
            f'Tasks left      | {"|".join(f" {round(val * self.sim.n_bots_per_goal):2d} " for val in obs_vec[30:38])}\n'
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

        sky_mask = (seg == cfg.ENT_CLS_SKY).unsqueeze(-1)
        rgb = torch.where(sky_mask, self.session.sky_clr, rgb.float())

        rgb = rgb.cpu().numpy()
        dep = 255. * norm_distance(-dep, cfg.MAX_IMG_DEPTH).cpu().numpy()
        seg = (255. / (cfg.N_ENT_CLASSES - 1)) * seg.cpu().numpy()

        self.gym.end_access_image_tensors(self.sim_handle)

        return rgb, dep, seg

    # --------------------------------------------------------------------------
    # MARK: get_visnet_images

    def get_visnet_images(self) -> 'tuple[np.ndarray, np.ndarray, np.ndarray]':
        obs_img = self.session.last_data['obs'][0]

        hsvd = obs_img[self.all_bot_idx:self.all_bot_idx+1]
        out = self.visnet(hsvd)

        clr_logits, ent_logits, dep = out.split(cfg.DEC_IMG_CHANNEL_SPLIT, dim=1)
        clr_probs = clr_logits[0].softmax(dim=0).cpu().numpy()
        ent_probs = ent_logits[0].softmax(dim=0).cpu().numpy()
        dep = dep[0, 0].cpu().numpy()

        dep = np.clip(dep * 255., 0., 255.)
        ent_seg = (ent_probs * self.ENT_CLASSES).sum(axis=0)
        # TODO: Remove slice after VisNet update
        rgb_seg = (clr_probs[:cfg.N_CLR_CLASSES, ..., None] * self.RGB_CLASSES).sum(axis=0)

        return rgb_seg, dep, ent_seg

    # --------------------------------------------------------------------------
    # MARK: save_camera_images

    def save_camera_images(self, reconstruct: bool = False) -> 'tuple[str, str, str]':
        rgb, dep, seg = (self.get_visnet_images if reconstruct else self.get_rendered_images)()

        # RGB
        file_idx = get_available_file_idx(cfg.DATA_DIR, 'img_rgbcam')
        filename_rgb = os.path.join(cfg.DATA_DIR, f'img_rgbcam_{file_idx:02d}.png')

        Image.fromarray(rgb.astype(np.uint8), mode='RGB').save(filename_rgb)

        # Depth
        file_idx = get_available_file_idx(cfg.DATA_DIR, 'img_depcam')
        filename_dep = os.path.join(cfg.DATA_DIR, f'img_depcam_{file_idx:02d}.png')

        Image.fromarray(dep.astype(np.uint8), mode='L').save(filename_dep)

        # Entity seg.
        file_idx = get_available_file_idx(cfg.DATA_DIR, 'img_segcam')
        filename_seg = os.path.join(cfg.DATA_DIR, f'img_segcam_{file_idx:02d}.png')

        Image.fromarray(seg.astype(np.uint8), mode='L').save(filename_seg)

        return filename_rgb, filename_dep, filename_seg

    # --------------------------------------------------------------------------
    # MARK: save_viewer_image

    def save_viewer_image(self) -> str:
        file_idx = get_available_file_idx(cfg.DATA_DIR, 'img_viewer')
        filename = os.path.join(cfg.DATA_DIR, f'img_viewer_{file_idx:02d}.png')

        self.gym.write_viewer_image_to_file(self.viewer, filename)

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

                # NOTE: Same as below
                elif cmd_key == 'cycle_env':
                    self.cycle_target_indices(env_inc=-1 if self.key_vec[-1] else 1)

                    print(f'Env/agent index switched to {self.env_idx}/{self.all_bot_idx}.')

                # NOTE: Previous bot torques are not automatically reset to zero
                elif cmd_key == 'cycle_bot':
                    self.cycle_target_indices(bot_inc=-1 if self.key_vec[-1] else 1)

                    print(f'Bot/agent index switched to {self.bot_idx}/{self.all_bot_idx}.')

                elif cmd_key == 'cycle_view':
                    self.view = (self.view + (-1 if self.key_vec[-1] else 1)) % 3
                    # self.set_top_view()

                    if self.view == self.VIEW_BOT:
                        self.gym.clear_lines(self.viewer)

                elif cmd_key == 'save_view':
                    if self.view != self.VIEW_TOP and self.session.render_cameras:
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

                elif cmd_key == 'light_beacon':
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

        self.update_view(update_lines=after_eval)
        self.gym.draw_viewer(self.viewer, self.sim_handle, False)
        self.gym.sync_frame_time(self.sim_handle)

    # --------------------------------------------------------------------------
    # MARK: reset

    def reset(self):
        self.key_vec.fill(0)
        self.update_view()


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

    REC_ALL = 2
    REC_VEC = 1
    REC_NONE = 0

    ARGS = [
        {'name': '--n_envs', 'type': int, 'default': 1, 'help': 'Number of parallel environments.'},
        {'name': '--n_bots', 'type': int, 'default': 2*cfg.N_GOAL_CLRS, 'help': 'Number of agents per environment.'},
        {'name': '--n_goals', 'type': int, 'default': -1, 'help': 'Number of goals per environment.'},
        {'name': '--global_spawn_prob', 'type': float, 'default': 0., 'help': 'Option to spawn bots across the maze.'},
        {'name': '--ep_duration', 'type': int, 'default': cfg.EP_DURATION, 'help': 'Episode duration in seconds.'},
        {'name': '--end_step', 'type': int, 'default': -1, 'help': 'Max steps until auto-termination.'},
        {'name': '--ctrl_mode', 'type': int, 'default': CTRL_MAN, 'help': 'Sim/agent control mode.'},
        {'name': '--rec_mode', 'type': int, 'default': REC_NONE, 'help': 'Data category to record.'},
        {'name': '--preview', 'type': int, 'default': 0, 'help': 'Option to view input or recons. images in side GUI.'},
        {'name': '--headless', 'type': int, 'default': 0, 'help': 'Option to run without a viewer.'},
        {'name': '--draw_freq', 'type': int, 'default': 64, 'help': 'Viewer frames per second.'},
        {'name': '--act_freq', 'type': int, 'default': cfg.STEPS_PER_SECOND, 'help': 'Inference steps per second.'},
        {'name': '--transfer_name', 'type': str, 'default': '', 'help': 'Starting model name/ID string.'},
        {'name': '--transfer_ver', 'type': int, 'default': -1, 'help': 'Starting model ckpt. version.'},
        {'name': '--model_name', 'type': str, 'default': 'mazeai', 'help': 'Model name/ID string.'},
        {'name': '--com_mode', 'type': int, 'default': cfg.COM_TOPIC, 'help': 'Mode of inter-agent communication.'},
        # TODO: Impl. auxiliary tasks
        {'name': '--aux_mode', 'type': int, 'default': cfg.AUX_NONE, 'help': 'Mode of auxiliary com. training.'},
        {'name': '--prob_actor', 'type': int, 'default': 1, 'help': 'Option to keep probabilistic inference.'},
        {'name': '--rng_seed', 'type': int, 'default': cfg.SEEDS[-1], 'help': 'Seed for numpy and torch RNGs.'}]

    def __init__(self, args: Namespace):
        self.end_step: int = args.end_step
        self.ctrl_mode: int = args.ctrl_mode
        self.rec_mode: int = args.rec_mode
        self.rec_data_queue: 'list[Tensor]' = []
        self.preview = bool(args.preview)

        # Resume model state
        self.model_options = {'n_envs': args.n_envs, 'n_bots': args.n_bots, 'com_mode': args.com_mode}
        self.prob_actor = bool(args.prob_actor)

        self.ckpter = CheckpointTracker(
            args.model_name, cfg.DATA_DIR, args.sim_device, args.rng_seed,
            transfer_name=args.transfer_name,
            ver_to_transfer=args.transfer_ver if args.transfer_ver >= 0 else None,
            reset_step_on_transfer=True)

        if self.ctrl_mode == self.CTRL_RL:
            self.ckpter.logger.info(f'Training with args.:\n{{{str(args)[10:-1]}}}')

        # Init. Isaac Gym, envs., and state tensors
        if args.headless:
            args.draw_freq = args.act_freq

        sim = MazeSim(args, self.ckpter.rng)
        interface = Interface(self, sim, args.sim_device) if not args.headless else None
        self.last_data = None

        super().__init__(
            sim,
            interface,
            args.ep_duration,
            args.act_freq,
            args.draw_freq,
            render_cameras=self.ctrl_mode > self.CTRL_MAN or self.rec_mode > self.REC_NONE or self.preview,
            keep_segmentation=self.rec_mode > self.REC_NONE,
            keep_rgb_over_hsv=self.ctrl_mode <= self.CTRL_GEN and not self.preview,
            stagger_env_resets=self.ctrl_mode == self.CTRL_RL,
            device=args.sim_device)

    # --------------------------------------------------------------------------
    # MARK: post_step

    def post_step(self, step_data: 'dict[str, Tensor | tuple[Tensor, ...]]') -> 'tuple[Tensor, ...]':
        obs = step_data['obs']

        # Keep data for debugging via interface
        self.last_data = step_data

        # Keep data to save
        if self.rec_mode:
            self.update_rec_data_queue(step_data)

            # Remove segmentation channel
            return (obs[0][:, :-1], *obs[1:])

        return obs

    # --------------------------------------------------------------------------
    # MARK: update_rec_data_queue

    def update_rec_data_queue(self, step_data: 'dict[str, Tensor | tuple[Tensor, ...]]'):
        img, vec, spa = step_data['obs']
        rwd = step_data['rwd']
        vaux = step_data['vaux']
        prio = step_data['prio']
        nrst = step_data['nrst']

        # Images are stored in full or vector form (as means)
        if self.rec_mode == self.REC_VEC:
            img = img.mean((-2, -1))
            spa = spa.mean((-2, -1))

        # TODO: Separate joint and individual reward
        # rwdj = rwdj.repeat_interleave(self.sim.n_bots, dim=0, output_size=self.sim.n_all_bots)
        prio = prio.repeat_interleave(self.sim.n_bots, dim=0, output_size=self.sim.n_all_bots)

        vecs = (vec, rwd, vaux, prio.unsqueeze(-1), nrst)

        img_data = img.cpu().numpy()
        vec_data = torch.hstack(vecs).cpu().numpy()
        map_data = spa.cpu().numpy()

        self.rec_data_queue.extend((img_data, vec_data, map_data))

    # --------------------------------------------------------------------------
    # MARK: save_rec_data

    def save_rec_data(self):
        if self.rec_mode == self.REC_NONE or not self.rec_data_queue:
            return

        file_idx = get_available_file_idx(cfg.DATA_DIR, 'rec')
        filename = os.path.join(cfg.DATA_DIR, f'rec_{file_idx:02d}.npz')

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

            print('Ending...')

        except KeyboardInterrupt:
            print('Interrupted...')

        self.save_rec_data()

        # Cleanup
        if self.interface is not None:
            self.gym.destroy_viewer(self.interface.viewer)

            if self.interface.tk_root is not None:
                self.interface.tk_root.destroy()

        self.sim.cleanup()
        self.async_event_loop.close()

        print('Done.')

    # --------------------------------------------------------------------------
    # MARK: train

    def train(self):
        encoder_path = os.path.join(cfg.ASSET_DIR, 'visenc.pt')

        # Load model
        model = ActorCritic(**self.model_options)

        model.to(self.ckpter.device)
        model.visencoder.load_state_dict(torch.load(encoder_path, map_location=self.ckpter.device))
        self.ckpter.load_model(model)

        # Init. optimizer and LR scheduler
        self.ckpter.optimizer = NAdamW(
            (param for param in model.parameters() if param.requires_grad),
            lr=1e-4,
            weight_decay=cfg.WEIGHT_DECAY,
            device=self.ckpter.device)

        scheduler = AnnealingScheduler(
            self.ckpter.optimizer,
            step_milestones=cfg.UPDATE_MILESTONE_MAP['policy'],
            starting_step=self.ckpter.meta['update_step'])

        entropy_scheduler = CoeffScheduler(
            cfg.UPDATE_MILESTONE_MAP['policy'][-1],
            cfg.ENT_WEIGHT_MILESTONES,
            starting_step=self.ckpter.meta['update_step'])

        rl_algo = PPG(
            self.step,
            self.ckpter,
            scheduler,
            self.sim.n_all_bots,
            cfg.N_EPOCHS_MAP['policy'],
            cfg.LOG_EPOCH_INTERVAL,
            cfg.CKPT_EPOCH_INTERVAL,
            cfg.BRANCH_EPOCH_INTERVAL,
            cfg.N_ROLLOUT_STEPS,
            cfg.N_TRUNCATED_STEPS,
            cfg.N_PASSES_PER_BATCH,
            None,
            cfg.N_ROLLOUTS_PER_EPOCH,
            cfg.N_AUX_ITERS_PER_EPOCH,
            cfg.DISCOUNTS,
            value_weight=cfg.VALUE_WEIGHT,
            aux_weight=cfg.AUX_WEIGHT,
            entropy_weight=entropy_scheduler,
            log_dir=cfg.LOG_DIR,
            accelerate=False)

        try:
            rl_algo.run()

        except KeyboardInterrupt:
            rl_algo.writer.close()
            raise

    # --------------------------------------------------------------------------
    # MARK: train_vis

    def train_vis(self):

        # TODO: Impl. auxiliary tasks
        raise NotImplementedError

    # --------------------------------------------------------------------------
    # MARK: eval

    def eval(self):
        if not self.headless:
            self.interface.set_top_view()

        model = ActorCritic(**self.model_options)

        model.to(self.ckpter.device)
        self.ckpter.load_model(model)

        mem = model.init_mem(self.sim.n_all_bots)

        with torch.inference_mode():
            obs = self.step(get_info=False)[0]['obs']
            step_ctr = -1

            while (step_ctr := step_ctr + 1) != self.end_step:
                act_sample, mem = model.act(obs, mem, sample=self.prob_actor)
                actions, beliefs = model.unwrap_sample(act_sample)

                step_data = self.step(actions, beliefs, get_info=False)[0]
                obs = self.post_step(step_data)

                if not step_data['nrst'].all():
                    mem = model.reset_mem(mem, step_data['nrst'])

    # --------------------------------------------------------------------------
    # MARK: test

    def test(self):
        if not self.headless:
            self.interface.set_top_view()

        with torch.inference_mode():
            self.step(get_info=False)
            step_ctr = -1

            while (step_ctr := step_ctr + 1) != self.end_step:
                self.post_step(self.step(self.actions, get_info=False)[0])


# --------------------------------------------------------------------------
# MARK: main

if __name__ == '__main__':
    args = gymutil.parse_arguments(description='Run MazeBots session.', custom_parameters=Session.ARGS)
    session = Session(args)
    session.run()
