"""Simulation control flow and runtime"""

import tkinter as tk
import os
from argparse import Namespace
from typing import Any, Callable

import numpy as np
from PIL import Image
from isaacgym import gymapi, gymutil
import torch
from torch import Tensor

from discit.accel import capture_graph
from discit.optim import AnnealingScheduler, CoeffScheduler, NAdamW
from discit.rl import PPG
from discit.track import CheckpointTracker

import config as cfg
from sim import MazeSim, CAM_OFFSET, MOT_MAX_TORQUE
from task import BasicInterface, MazeTask, MAX_IMG_DEPTH, TIME_SCALE
from model import ActorCritic, Policy, VisNet
from utils import get_available_file_idx
from utils_torch import apply_quat_rot, get_eulz_from_quat, norm_distance


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

    OFFSET_BOT_AHEAD = [[CAM_OFFSET[0], 0., 0.]]
    OFFSET_BOT_ABOVE = [[0., 0., CAM_OFFSET[2]]]

    OFFSET_3RD_AHEAD = [[-cfg.BOT_WIDTH*2, 0., 0.]]
    OFFSET_3RD_ABOVE = [[0., 0., cfg.WALL_HALFHEIGHT]]

    OBJ_LINE_NUM = 16
    OBJ_LINE_OFFSET = 0.01
    OBJ_LINE_CLR = (1., 0., 0.)

    DIR_LINE_OFFSET = 0.01
    DIR_LINE_LENGTH = cfg.BOT_RADIUS * 2.
    DIR_LINE_CLR_DEFAULT = (1., 0., 0.)
    DIR_LINE_CLR_GOAL_SEEN = (0., 1., 0.)
    DIR_LINE_CLR_OBJ_SEEN = (0., 0., 1.)

    PREV_ZOOM = 9
    PREV_DIM = (cfg.OBS_IMG_RES_WIDTH, 3 * cfg.OBS_IMG_RES_HEIGHT)

    TYP_CLASSES = np.linspace(0, 255, 7, dtype=np.float32)[:, None, None]
    RGB_CLASSES = np.round(np.array(cfg.ALL_CLR_CLASSES, dtype=np.float32)[:, None, None] * 255.)
    RGB_CLASSES[-2] = 168.  # Override 127, as the ground appears brighter in renders

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

    def __init__(self, session: 'Session', sim: MazeSim, device: str):
        super().__init__(sim.gym, sim.handle)

        self.session = session
        self.sim = sim

        # Init viewer
        viewer_top_offset = sim.constructor.supenv_halfwidth * 1.5
        self.viewer_top_pos = gymapi.Vec3(0., viewer_top_offset, viewer_top_offset)
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

        self.obj_line_num = sim.n_objects * self.OBJ_LINE_NUM
        obj_line_angles = np.array([i*2.*np.pi/self.OBJ_LINE_NUM for i in range(self.OBJ_LINE_NUM)], dtype=np.float32)
        obj_line_pts = np.stack((np.cos(obj_line_angles), np.sin(obj_line_angles)), axis=-1) * cfg.GOAL_RADIUS
        self.obj_line_start_pts = np.tile(obj_line_pts, (sim.n_objects, 1))
        self.obj_line_end_pts = np.concatenate((self.obj_line_start_pts[1:], self.obj_line_start_pts[:1]))
        self.obj_line_offset = np.array((self.OBJ_LINE_OFFSET,)*self.obj_line_num, dtype=np.float32)[:, None]
        self.obj_line_clr = np.array((self.OBJ_LINE_CLR,)*self.obj_line_num, dtype=np.float32)

        self.dir_line_offset = np.array((self.DIR_LINE_OFFSET,)*sim.n_bots, dtype=np.float32)[:, None]
        self.dir_line_clr_dft = torch.tensor((self.DIR_LINE_CLR_DEFAULT,)*sim.n_bots, dtype=torch.float32)
        self.dir_line_clr_goal = torch.tensor((self.DIR_LINE_CLR_GOAL_SEEN,)*sim.n_bots, dtype=torch.float32)
        self.dir_line_clr_obj = torch.tensor((self.DIR_LINE_CLR_OBJ_SEEN,)*sim.n_bots, dtype=torch.float32)

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

    def cycle_target_indices(self, env_inc: int = 0, bot_inc: int = 0):
        self.env_idx = (self.env_idx + env_inc) % self.sim.n_envs
        self.bot_idx = (self.bot_idx + bot_inc) % self.sim.n_bots
        self.all_bot_idx = self.env_idx * self.sim.n_bots + self.bot_idx

    def set_top_view(self):
        if self.view != self.VIEW_TOP:
            return

        self.gym.viewer_camera_look_at(
            self.viewer,
            self.sim.envs[self.env_idx].handle,
            self.viewer_top_pos,
            self.viewer_top_target)

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

    def update_dir_lines(self):
        tmp = self.session.async_temp_result

        if tmp is None:
            return

        obs_vec = tmp[0 if len(tmp[0].shape) == 2 else 1]
        obs_vec = obs_vec[self.env_idx*self.sim.n_bots:(self.env_idx+1)*self.sim.n_bots, 27:40].cpu()

        obj_in_sight = obs_vec[:, :9].any(-1, keepdim=True)
        goal_in_sight = obs_vec[:, 9:10]

        # TODO: Show the agents' internal dir. estimation
        # if self.prev_reconstruct ...
        dir_vec = obs_vec[:, -3:]

        bot_ori = self.session.env_bot_ori[self.env_idx].cpu()
        dir_vec = apply_quat_rot(bot_ori, dir_vec)

        goal_dir = (dir_vec[:, :2] * self.DIR_LINE_LENGTH).numpy()
        goal_prox = dir_vec[:, 2:]

        dir_line_clr = torch.where(
            obj_in_sight,
            torch.lerp(self.dir_line_clr_obj, self.dir_line_clr_goal, goal_in_sight),
            self.dir_line_clr_dft).mul_(goal_prox).numpy()

        self.gym.add_lines(
            self.viewer,
            self.sim.envs[self.env_idx].handle,
            self.sim.n_bots,
            np.concatenate((
                self.session.bot_pos_arr, self.dir_line_offset,
                self.session.bot_pos_arr + goal_dir, self.dir_line_offset), axis=-1),
            dir_line_clr)

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

    def get_torque_from_key_vec(self) -> Tensor:
        """Compute 4-wheel torques from keyboard press state."""

        mvmt_forw = self.key_vec[0:2].sum()
        mvmt_left = self.key_vec[2:4].sum()
        scale = (MOT_MAX_TORQUE / 2.) if self.key_vec[4] else MOT_MAX_TORQUE

        if not (mvmt_forw or mvmt_left):
            return self.torque_states[np.nan]

        return self.torque_states.get(np.arctan2(mvmt_left, mvmt_forw), self.torque_states[np.nan]) * scale

    def get_debug_info(self) -> str:
        if self.session.async_temp_result is None:
            return 'State not yet evaluated.\n'

        obs_img, obs_vec, obs_spa = self.session.async_temp_result
        idx, idy = torch.bucketize(self.session.bot_pos[self.all_bot_idx], self.session.cell_delims)

        obs_img = obs_img[self.all_bot_idx].cpu().mean((-2, -1)).numpy()
        obs_vec = obs_vec[self.all_bot_idx].cpu().numpy()
        obs_spa = obs_spa[self.env_idx, :, idx, idy].cpu().numpy()

        obs_com = obs_vec[44:94].reshape(10, 5)
        rgb_arr = np.array(cfg.RCVR_CLR_CLASSES)
        act_rgb = rgb_arr[np.argmax(obs_vec[33:44])]

        goal_com = obs_com[self.session.goal_idx[self.all_bot_idx].item() + 1, :4]
        obs_com = obs_com[:, None, :4] * rgb_arr[1:, :, None] * obs_com[:, None, 4:]
        obs_com = obs_com.sum(0).flatten()

        time_left = (self.sim.ep_duration - self.session.env_run_times[self.env_idx]).item()
        score = self.session.get_score()
        bot_ori = self.session.env_bot_ori[None, self.env_idx, self.bot_idx].cpu()
        z_angle = get_eulz_from_quat(bot_ori).item() * 180. / np.pi
        cell_exp = self.session.cell_exploration[self.all_bot_idx, idx, idy].item()

        return (
            '\nSESSION\n'
            f'Time to ep. end | {max(0., time_left): .2f}s\n'
            f'Avg. score      | {score: .2f} (per bot)\n'
            f'Ori. angle (z)  | {z_angle: .0f}\n'
            f'Explor. bonus   | {cell_exp: .0f}\n\n'

            'GUIDE\n'
            f'Goal com. chn.  |                Front: {goal_com[0]: .2f}\n'
            f'                | Left:  {goal_com[3]: .2f} | Back:  {goal_com[2]: .2f} | Right: {goal_com[1]: .2f}\n'
            f'Goal com. dir.  | Front: {goal_com[0] - goal_com[2]: .2f} | Left: {goal_com[3] - goal_com[1]: .2f}\n'
            f'Air direction   | Front: {obs_vec[131]: .2f} | Left: {obs_vec[132]: .2f}\n'
            f'A*  direction   | Front: {obs_vec[134]: .2f} | Left: {obs_vec[135]: .2f}\n'
            f'Air proximity   |        {obs_vec[133]: .2f}\n'
            f'A*  proximity   |        {obs_vec[136]: .2f}\n'
            f'Goal position   | X:     {obs_vec[137]: .2f} | Y:    {obs_vec[138]: .2f}\n'
            f'Goal in sight   | {"TRUE" if obs_vec[94] else "FALSE"}\n'
            f'Obj. in sight   | {"|".join(" X " if obj_in_frame else " _ " for obj_in_frame in obs_vec[95:104])}\n'
            f'Obj. proximity  | {"|".join(f"{obj_prox:.1f}" for obj_prox in 10*obs_vec[122:131])}\n\n'

            'INTERACTION\n'
            f'Contact flag    | {"TRUE" if obs_vec[149] else "FALSE"}\n\n'

            'TASK\n'
            f'Spec. index     | {np.argmax(obs_vec[21:30])}\n'
            f'Speaking role   | {"TRUE" if obs_vec[30] else "FALSE"}\n'
            f'Time at goal    | {obs_vec[31]: .2f}s\n'
            f'Own throughput  | {obs_vec[32]: .2f} (per 30s)\n'
            f'Avg. throughput | {obs_vec[151]: .2f} (per bot per 30s)\n'
            f'Time to ep. end | {obs_vec[161] / TIME_SCALE: .2f}s\n'
            f'Time on task    | {obs_vec[141] / TIME_SCALE: .2f}s\n'
            f'New/done tasks  | {obs_vec[140]: .0f}\n'
            f'Dist. diff.     | {obs_vec[139]: .2f}m\n\n'

            'OBSERVATION\n'
            f'Avg. img. chan. | R: {obs_img[0]: .2f} | G: {obs_img[1]: .2f} | B: {obs_img[2]: .2f}\n'
            f'                | D: {obs_img[3]: .2f}\n'
            f'Act. torques    | FL: {obs_vec[0]: .2f} | FR: {obs_vec[1]: .2f}\n'
            f'                | BL: {obs_vec[2]: .2f} | BR: {obs_vec[3]: .2f}\n'
            f'IMU ang. vel.   | X: {obs_vec[4]: .2f} | Y: {obs_vec[5]: .2f} | Z: {obs_vec[6]: .2f}\n'
            f'IMU accel.      | X: {obs_vec[7]: .2f} | Y: {obs_vec[8]: .2f} | Z: {obs_vec[9]: .2f}\n'
            f'IMU magnet.     | X: {obs_vec[10]: .2f} | Y: {obs_vec[11]: .2f} | Z: {obs_vec[12]: .2f}\n'
            f'AHRS ori. quat. | X: {obs_vec[13]: .2f} | Y: {obs_vec[14]: .2f} | Z: {obs_vec[15]: .2f}\n'
            f'                | W: {obs_vec[16]: .2f}\n'
            f'GPS bot pos.    | X: {obs_vec[17]: .2f} | Y: {obs_vec[18]: .2f}\n'
            f'GPS bot vel.    | X: {obs_vec[19]: .2f} | Y: {obs_vec[20]: .2f}\n'
            f'Act. colour     | R: {act_rgb[0]: .2f} | G: {act_rgb[1]: .2f} | B: {act_rgb[2]: .2f}\n\n'
            f'RGB rcvr. Front | R: {obs_com[0]: .2f} | G: {obs_com[4]: .2f} | B: {obs_com[8]: .2f}\n'
            f'RGB rcvr. Right | R: {obs_com[1]: .2f} | G: {obs_com[5]: .2f} | B: {obs_com[9]: .2f}\n'
            f'RGB rcvr. Back  | R: {obs_com[2]: .2f} | G: {obs_com[6]: .2f} | B: {obs_com[10]: .2f}\n'
            f'RGB rcvr. Left  | R: {obs_com[3]: .2f} | G: {obs_com[7]: .2f} | B: {obs_com[11]: .2f}\n')

    def get_rendered_images(self) -> 'tuple[np.ndarray, np.ndarray, np.ndarray]':
        self.gym.render_all_camera_sensors(self.sim_handle)
        self.gym.start_access_image_tensors(self.sim_handle)

        rgb = self.session.img_rgb_list[self.all_bot_idx][..., :3]
        dep = self.session.img_dep_list[self.all_bot_idx]
        typ = self.session.img_seg_list[self.all_bot_idx]

        if self.sim.is_preset:
            sky_mask = (typ == cfg.SEG_CLS_NULL).unsqueeze(-1)
            rgb = torch.where(sky_mask, self.session.sky_clr, rgb.float())

        rgb = rgb.cpu().numpy()
        dep = 255. * norm_distance(-dep, MAX_IMG_DEPTH).cpu().numpy()
        typ = (255. / 6.) * typ.cpu().numpy()

        self.gym.end_access_image_tensors(self.sim_handle)

        return rgb, dep, typ

    def get_visnet_images(self) -> 'tuple[np.ndarray, np.ndarray, np.ndarray]':
        obs_img, _, _ = self.session.async_temp_result

        hsvd = obs_img[self.all_bot_idx:self.all_bot_idx+1]

        out = self.visnet(hsvd)

        clr_logits, typ_logits, dep = out.split(cfg.DEC_IMG_CHANNEL_SPLIT, dim=1)
        clr_probs = clr_logits[0].softmax(dim=0).cpu().numpy()
        typ_probs = typ_logits[0].softmax(dim=0).cpu().numpy()
        dep = dep[0, 0].cpu().numpy()

        dep = np.clip(dep * 255., 0., 255.)
        typ_seg = (typ_probs * self.TYP_CLASSES).sum(axis=0)
        rgb_cls = (clr_probs[..., None] * self.RGB_CLASSES).sum(axis=0)

        return rgb_cls, dep, typ_seg

    def save_camera_images(self, reconstruct: bool = False) -> 'tuple[str, str, str]':
        rgb, dep, typ = (self.get_visnet_images if reconstruct else self.get_rendered_images)()

        # RGB
        file_idx = get_available_file_idx(cfg.DATA_DIR, 'img_rgbcam')
        filename_rgb = os.path.join(cfg.DATA_DIR, f'img_rgbcam_{file_idx:02d}.png')

        Image.fromarray(rgb.astype(np.uint8), mode='RGB').save(filename_rgb)

        # Depth
        file_idx = get_available_file_idx(cfg.DATA_DIR, 'img_depcam')
        filename_dep = os.path.join(cfg.DATA_DIR, f'img_depcam_{file_idx:02d}.png')

        Image.fromarray(dep.astype(np.uint8), mode='L').save(filename_dep)

        # Type seg.
        file_idx = get_available_file_idx(cfg.DATA_DIR, 'img_typcam')
        filename_typ = os.path.join(cfg.DATA_DIR, f'img_typcam_{file_idx:02d}.png')

        Image.fromarray(typ.astype(np.uint8), mode='L').save(filename_typ)

        return filename_rgb, filename_dep, filename_typ

    def save_viewer_image(self) -> str:
        file_idx = get_available_file_idx(cfg.DATA_DIR, 'img_viewer')
        filename = os.path.join(cfg.DATA_DIR, f'img_viewer_{file_idx:02d}.png')

        self.gym.write_viewer_image_to_file(self.viewer, filename)

        return filename

    def eval_events(self):
        """Check for keyboard events, update torque, colours, and view."""

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
                            filename_rgb, filename_dep, filename_typ = self.save_camera_images(reconstruct)

                            print(
                                f'Saved camera RGB image to: {filename_rgb}\n'
                                f'Saved camera DEP image to: {filename_dep}\n'
                                f'Saved camera TYP image to: {filename_typ}')

                    print(f'Saved viewer image to: {self.save_viewer_image()}')

            elif cmd_key in self.ACT_KEYS:
                if cmd_key == 'alt_move':
                    self.key_vec[-1] = event.value

                    if not event.value:
                        self.prev_reconstruct = not self.prev_reconstruct

                        print(f'Side preview reconstruction is {"ON" if self.prev_reconstruct else "OFF"}.')

                elif self.session.ctrl_mode != Session.CTRL_MAN:
                    if cmd_press:
                        print('Cannot command agents without manual control mode.')

                    continue

                elif cmd_key == 'recolour':
                    if cmd_press:
                        self.session.actions[:, -cfg.N_DIM_RGB:] = self.session.sample_colours()

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

    def update_preview(self):
        if not self.session.preview or self.session.async_temp_result is None:
            return

        rgb, dep, typ = (self.get_visnet_images if self.prev_reconstruct else self.get_rendered_images)()
        stacked = np.concatenate((
            rgb, np.broadcast_to(dep[..., None], rgb.shape), np.broadcast_to(typ[..., None], rgb.shape)), axis=0)

        data = f'P6 {self.PREV_DIM[0]} {self.PREV_DIM[1]} 255 '.encode() + stacked.astype(np.uint8).tobytes()

        photoimage = tk.PhotoImage(
            width=self.PREV_DIM[0],
            height=self.PREV_DIM[1],
            data=data,
            format='PPM'
        ).zoom(self.PREV_ZOOM, self.PREV_ZOOM)

        self.tk_canvas.create_image(0, 0, anchor='nw', image=photoimage)
        self.tk_root.update()

    def sync_redraw(self, after_eval: bool = True):
        """Draw the scene in the viewer, syncing sim with real-time."""

        self.update_view(update_lines=after_eval)
        self.gym.draw_viewer(self.viewer, self.sim_handle, False)
        self.gym.sync_frame_time(self.sim_handle)

    def reset(self):
        self.key_vec.fill(0)
        self.update_view()


class Session(MazeTask):
    """
    The main process descriptor connecting environment, operational,
    and training constructs, and providing several interfacing options.
    """

    CTRL_AI = 3
    CTRL_RL = 2
    CTRL_GEN = 1
    CTRL_MAN = 0

    REC_ALL = 3
    REC_IMG = 2
    REC_VEC = 1
    REC_NONE = 0

    ARGS = [
        {'name': '--level', 'type': int, 'default': 4, 'help': 'Maze complexity level.'},
        {'name': '--preset_name', 'type': str, 'default': 'env7', 'help': 'Name of files with preset level data.'},
        {'name': '--regen', 'type': int, 'default': 0, 'help': 'Option to fully regenerate environments on reset.'},
        {'name': '--n_bots', 'type': int, 'default': -1, 'help': 'Number of agents per environment.'},
        {'name': '--n_speakers', 'type': int, 'default': -1, 'help': 'Number of communicating agents per environment.'},
        {'name': '--n_envs', 'type': int, 'default': -1, 'help': 'Number of parallel environments.'},
        {'name': '--mul_duration', 'type': float, 'default': 1., 'help': 'Episode duration multiplier.'},
        {'name': '--end_step', 'type': int, 'default': -1, 'help': 'Max steps until auto-termination.'},
        {'name': '--ctrl_mode', 'type': int, 'default': CTRL_MAN, 'help': 'Sim/agent control mode.'},
        {'name': '--rec_mode', 'type': int, 'default': REC_NONE, 'help': 'Data category to record.'},
        {'name': '--preview', 'type': int, 'default': 0, 'help': 'Option to view input or recons. images in side GUI.'},
        {'name': '--headless', 'type': int, 'default': 0, 'help': 'Option to run without a viewer.'},
        {'name': '--act_freq', 'type': int, 'default': cfg.STEPS_PER_SECOND, 'help': 'Inference steps per second.'},
        {'name': '--transfer_name', 'type': str, 'default': '', 'help': 'Starting model name/ID string.'},
        {'name': '--transfer_ver', 'type': int, 'default': -1, 'help': 'Starting model ckpt. version.'},
        {'name': '--model_name', 'type': str, 'default': 'mazeai', 'help': 'Model name/ID string.'},
        {'name': '--com_state', 'type': int, 'default': Policy.FEAT_TRAINED, 'help': 'St. of inter-ag. communication.'},
        {'name': '--guide_state', 'type': int, 'default': Policy.FEAT_DISABLED, 'help': 'St. of directional guidance.'},
        {'name': '--com_bias', 'type': int, 'default': 0, 'help': 'Option to bias twd. unassociated com.'},
        {'name': '--prob_actor', 'type': int, 'default': 1, 'help': 'Option to keep probabilistic inference.'},
        {'name': '--team_reward', 'type': int, 'default': 1, 'help': 'Final group score instead of individ. rewards.'},
        {'name': '--rng_seed', 'type': int, 'default': 42, 'help': 'Seed for numpy and torch RNGs.'},
        {'name': '--schedule_key', 'type': str, 'default': '', 'help': 'Key of a training curriculum stage.'}]

    def __init__(self, args: Namespace):
        self.end_step: int = args.end_step
        self.ctrl_mode: int = args.ctrl_mode
        self.rec_mode: int = args.rec_mode
        self.rec_data_queue: 'list[Tensor]' = []
        self.preview = bool(args.preview)

        # Resume model state
        self.model_options = {
            'com_state': args.com_state,
            'guide_state': args.guide_state,
            'com_bias': bool(args.com_bias)}

        self.prob_actor = bool(args.prob_actor)

        if not args.schedule_key:
            self.schedule_key = '128e-2a-15s'

        elif args.schedule_key not in cfg.TIME_MILESTONE_MAP:
            raise KeyError(f'Unknown schedule: {args.schedule_key}')

        else:
            self.schedule_key = args.schedule_key

        self.frozen_actor = self.schedule_key.split('-')[1][-1] != 'a'

        self.ckpter = CheckpointTracker(
            args.model_name, cfg.DATA_DIR, args.sim_device, args.rng_seed,
            transfer_name=args.transfer_name,
            ver_to_transfer=args.transfer_ver if args.transfer_ver >= 0 else None,
            reset_step_on_transfer=True)

        # Init IsaacGym and generate initial envs
        self.steps_per_second: int = min(args.act_freq, 64)
        frames_per_second = self.steps_per_second if args.headless else 64

        preset_path = os.path.join(cfg.ASSET_DIR, args.preset_name) if args.preset_name else None

        sim = MazeSim(args.level, args.n_bots, args.n_envs, frames_per_second, args, self.ckpter.rng, preset_path)
        interface = Interface(self, sim, args.sim_device) if not args.headless else None

        # Extend or diminish standard episode duration
        sim.ep_duration: int = round(sim.ep_duration * args.mul_duration)

        # Prepare computational graph components
        super().__init__(
            sim,
            interface,
            self.steps_per_second,
            frames_per_second,
            render_cameras=self.ctrl_mode > self.CTRL_MAN or self.rec_mode > self.REC_NONE or self.preview,
            keep_segmentation=self.rec_mode > self.REC_NONE,
            keep_rgb_over_hsv=self.ctrl_mode == self.CTRL_GEN,
            spawn_with_random_rgb=self.ctrl_mode == self.CTRL_GEN,
            uniform_task_sampling=self.ctrl_mode == self.CTRL_GEN,
            distribute_env_resets=self.ctrl_mode == self.CTRL_RL,
            full_env_regeneration=preset_path is None and args.regen,
            long_range_obj_signal=args.guide_state == Policy.FEAT_EXTERNAL,
            use_team_reward=bool(args.team_reward),
            num_speakers_per_env=args.n_speakers if args.n_speakers >= 0 else None,
            device=args.sim_device)

        # self.accelerate()

    def post_step(
        self,
        obs: 'tuple[Tensor, ...]',
        reward: Tensor,
        rst_mask_f: Tensor,
        *_other: 'tuple[Any, ...]'
    ) -> 'tuple[Tensor, ...]':

        # Keep data for debugging via interface
        self.async_temp_result = obs

        # Keep data to save
        if self.rec_mode:
            self.update_rec_data_queue(obs, reward, rst_mask_f)

            # Remove segmentation channel
            return (obs[0][:, :-1], *obs[1:])

        return obs

    def update_rec_data_queue(self, obs: 'tuple[Tensor, ...]', rew: Tensor, rst: Tensor):
        img, vec, spa = obs

        # Images are stored in full or vector form (as means)
        if self.rec_mode == self.REC_VEC:
            img = img.mean((-2, -1))
            spa = spa.mean((-2, -1))

        vecs = (vec, rew.unsqueeze(-1), rst.unsqueeze(-1))

        img_data = img.cpu().numpy()
        vec_data = torch.hstack(vecs).cpu().numpy()
        spa_data = spa.cpu().numpy()

        self.rec_data_queue.extend((img_data, vec_data, spa_data))

    def save_rec_data(self):
        if self.rec_mode == self.REC_NONE or not self.rec_data_queue:
            return

        file_idx = get_available_file_idx(cfg.DATA_DIR, 'rec')
        filename = os.path.join(cfg.DATA_DIR, f'rec_{file_idx:02d}.npz')

        img = np.stack(self.rec_data_queue[0::3])
        vec = np.stack(self.rec_data_queue[1::3])
        spa = np.stack(self.rec_data_queue[2::3])
        self.rec_data_queue.clear()

        np.savez_compressed(filename, img=img, vec=vec, spa=spa)

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

            if self.interface.tk_root is not None:
                self.interface.tk_root.destroy()

        self.sim.cleanup()
        self.async_event_loop.close()

        print('Done.')

    def train(self):
        encoder_path = os.path.join(cfg.ASSET_DIR, 'visenc.pt')

        # Load model
        model = ActorCritic(self.sim.n_bots, self.sim.n_envs, **self.model_options)

        optimizer = NAdamW(
            (param for param in model.parameters() if param.requires_grad),
            lr=1e-4,
            weight_decay=cfg.WEIGHT_DECAY)

        model.to(self.ckpter.device)
        model.visencoder.load_state_dict(torch.load(encoder_path, map_location=self.ckpter.device))

        if self.frozen_actor:
            self.ckpter.load_model(model)
            self.ckpter.optimizer = optimizer
            model.valuator.random_init()

        else:
            self.ckpter.load_model(model, optimizer)

        # Init. LR scheduler
        scheduler = AnnealingScheduler(
            optimizer,
            step_milestones=cfg.UPDATE_MILESTONE_MAP[self.schedule_key],
            starting_step=self.ckpter.meta['update_step'])

        # Accelerate collector & recollector
        # mem = model.init_mem(self.sim.n_all_bots)
        # model.collect_static = self.accel_action(model.collect_static, mem)
        # model.collect_copied = self.accel_action(model.collect_copied, mem, encode=False)

        # Assemble discount factors wrt. different rewards
        gammas = (
            0.5 ** (1. / (60 * self.steps_per_second)),     # Long-term (main) rewards: Half-life at 1 minute
            0.5 ** (1. / (3 * self.steps_per_second)))      # Short-term (aux.) rewards: Half-life at 3 seconds

        entropy_scheduler = CoeffScheduler(
            cfg.UPDATE_MILESTONE_MAP[self.schedule_key][-1],
            cfg.ENT_WEIGHT_MILESTONES,
            starting_step=self.ckpter.meta['update_step'])

        rl_algo = PPG(
            self.step,
            self.ckpter,
            scheduler,
            self.sim.n_all_bots,
            cfg.N_EPOCHS_MAP[self.schedule_key],
            cfg.LOG_EPOCH_INTERVAL,
            cfg.CKPT_EPOCH_INTERVAL,
            cfg.BRANCH_EPOCH_INTERVAL,
            cfg.N_ROLLOUT_STEPS,
            cfg.N_TRUNCATED_STEPS,
            cfg.N_PASSES_PER_BATCH,
            None,
            cfg.N_ROLLOUTS_PER_EPOCH,
            cfg.N_AUX_ITERS_PER_EPOCH,
            gammas,
            policy_weight=float(not self.frozen_actor),
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

    def eval(self):
        if not self.headless:
            self.interface.set_top_view()

        model = ActorCritic(self.sim.n_bots, self.sim.n_envs, **self.model_options)

        model.to(self.ckpter.device)
        self.ckpter.load_model(model)

        # Accelerate actor
        mem = model.init_mem(self.sim.n_all_bots)
        # model.act_partial = self.accel_action(model.act_partial, mem)

        with torch.inference_mode():
            obs = self.step(get_info=False)[0]
            step_ctr = -1

            while (step_ctr := step_ctr + 1) != self.end_step:
                act_sample, mem = model.act(obs, mem, sample=self.prob_actor)
                actions, action_indices = model.unwrap_sample(act_sample)

                obs, reward, rst_mask_f, *_ = self.step(actions, action_indices, get_info=False)
                obs = self.post_step(obs, reward, rst_mask_f)

                if not rst_mask_f.all():
                    mem = model.reset_mem(mem, 1. - rst_mask_f)

    def play(self):
        if not self.headless:
            self.interface.set_top_view()

        with torch.inference_mode():
            self.step(get_info=False)
            step_ctr = -1

            while (step_ctr := step_ctr + 1) != self.end_step:
                self.post_step(*self.step(self.actions, get_info=False))

    def accel_action(
        self,
        act_fn: Callable,
        aux_tensors: 'tuple[Tensor, ...]' = None,
        encode: bool = True
    ) -> Callable:

        if aux_tensors is None:
            aux_tensors = ()

        # Graph 3
        if encode:
            inputs = (
                self.graphs['prepare_images']['out'] if self.render_cameras else self.null_obs_img,
                *self.graphs['eval_state']['out'][:2],
                *[aux.detach().clone() for aux in aux_tensors])

            act_fn, self.graphs['act_partial'] = \
                capture_graph(act_fn, inputs, copy_idcs_in=tuple(range(len(inputs)-len(aux_tensors), len(inputs))))

        # Graph 4
        else:
            inputs = (
                torch.rand_like(self.graphs['act_partial']['out'][2]),
                torch.rand_like(self.graphs['act_partial']['out'][3]),
                self.graphs['eval_state']['out'][1].detach().clone(),
                *[aux.detach().clone() for aux in aux_tensors])

            act_fn, self.graphs['act_partial_encoded'] = capture_graph(act_fn, inputs)

        return act_fn


if __name__ == '__main__':
    args = gymutil.parse_arguments(description='Run MazeBots session.', custom_parameters=Session.ARGS)
    session = Session(args)
    session.run()
