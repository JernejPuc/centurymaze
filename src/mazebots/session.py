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
from discit.optim import NAdamW, AdaptivePlateauScheduler
from discit.rl import PPG
from discit.track import CheckpointTracker

import config as cfg
from sim import MazeSim, CAM_OFFSET, MOT_MAX_TORQUE
from task import BasicInterface, MazeTask, MAX_IMG_DEPTH, SCALE_TIME
from model import ActorCritic, Policy, VisNet
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

    OFFSET_BOT_AHEAD = [[CAM_OFFSET[0], 0., 0.]]
    OFFSET_BOT_ABOVE = [[0., 0., CAM_OFFSET[2]]]

    OFFSET_3RD_AHEAD = [[-cfg.BOT_WIDTH*2, 0., 0.]]
    OFFSET_3RD_ABOVE = [[0., 0., cfg.WALL_HALFHEIGHT]]

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

        # Side panel
        self.prev_reconstruct = False

        if session.preview:
            self.tk_root = tk.Tk()
            self.tk_canvas = tk.Canvas(
                self.tk_root,
                width=self.PREV_DIM[0] * self.PREV_ZOOM,
                height=self.PREV_DIM[1] * self.PREV_ZOOM)

            self.tk_canvas.pack()

            self.visnet = VisNet().to(session.ckpter.device)
            self.visnet.load_state_dict(torch.load(os.path.join(cfg.ASSET_DIR, 'visnet.pt')))

        else:
            self.tk_root = self.tk_canvas = self.visnet = None

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
        scale = (MOT_MAX_TORQUE / 2.) if self.key_vec[4] else MOT_MAX_TORQUE

        if not (mvmt_forw or mvmt_left):
            return self.torque_states[np.nan]

        return self.torque_states.get(np.arctan2(mvmt_left, mvmt_forw), self.torque_states[np.nan]) * scale

    def get_debug_info(self) -> str:
        if self.session.async_temp_result is None:
            return 'State not yet evaluated.\n'

        obs_img, obs_vec, com_weights = self.session.async_temp_result
        obs_com = Policy.weigh_signals(self.sim, obs_vec[:, cfg.OBS_RGB_SLICE], com_weights)

        obs_img = obs_img[self.all_bot_idx].cpu().mean((-2, -1)).numpy()
        obs_vec = obs_vec[self.all_bot_idx].cpu().numpy()
        obs_com = obs_com[self.all_bot_idx].cpu().flatten().numpy()

        # TODO: Reindex and handle omitted
        goal_in_sight = obs_vec[35]
        obs_vec = np.concatenate((obs_vec[2:26], obs_vec[36:51], obs_vec[54:]))

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
            f'Goal position   | Front: {obs_vec[24]: .2f} | Left: {obs_vec[25]: .2f}\n'
            f'Bot position    | Front: {obs_vec[26]: .2f} | Left: {obs_vec[27]: .2f}\n'
            f'Air direction   | Front: {obs_vec[28]: .2f} | Left: {obs_vec[29]: .2f}\n'
            f'A*  direction   | Front: {obs_vec[31]: .2f} | Left: {obs_vec[32]: .2f}\n'
            f'Air proximity   |        {obs_vec[30]: .2f}\n'
            f'A*  proximity   |        {obs_vec[33]: .2f}\n'
            f'Goal in sight   | {"TRUE" if goal_in_sight else "FALSE"}\n\n'

            'OBSTRUCTION\n'
            f'Prox. channels  | Front: {obs_vec[34]: .2f} | Right: {obs_vec[35]: .2f}\n'
            f'                | Back:  {obs_vec[36]: .2f} | Left:  {obs_vec[37]: .2f}\n'
            f'Contact flag    | {"TRUE" if obs_vec[38] else "FALSE"}\n\n'

            'TASK\n'
            f'Time at goal    | {obs_vec[39]: .2f}s\n'
            f'Time on task    | {obs_vec[40] / SCALE_TIME: .2f}s\n'
            f'Time to ep. end | {obs_vec[41] / SCALE_TIME: .2f}s\n'
            f'New/done tasks  | {obs_vec[42]: .0f}\n'
            f'Avg. throughput | {obs_vec[43]: .2f} (per bot per 10s)\n\n'

            'OBSERVATION\n'
            f'Avg. img. chan. | R: {obs_img[0]: .2f} | G: {obs_img[1]: .2f} | B: {obs_img[2]: .2f}\n'
            f'                | D: {obs_img[3]: .2f}\n'
            f'Act. torques    | {obs_vec[1]: .2f}, {obs_vec[2]: .2f}, {obs_vec[3]: .2f}, {obs_vec[4]: .2f}\n'
            f'DOF vel.        | {obs_vec[5]: .2f}, {obs_vec[6]: .2f}, {obs_vec[7]: .2f}, {obs_vec[8]: .2f}\n'
            f'IMU ang. vel.   | X: {obs_vec[9]: .2f} | Y: {obs_vec[10]: .2f} | Z: {obs_vec[11]: .2f}\n'
            f'IMU accel.      | X: {obs_vec[12]: .2f} | Y: {obs_vec[13]: .2f} | Z: {obs_vec[14]: .2f}\n'
            f'IMU magnet.     | X: {obs_vec[15]: .2f} | Y: {obs_vec[16]: .2f} | Z: {obs_vec[17]: .2f}\n'
            f'Goal colour     | R: {obs_vec[18]: .2f} | G: {obs_vec[19]: .2f} | B: {obs_vec[20]: .2f}\n'
            f'Act. colour     | R: {obs_vec[21]: .2f} | G: {obs_vec[22]: .2f} | B: {obs_vec[23]: .2f}\n\n'
            f'RGB rcvr. Front | R: {obs_com[0]: .2f} | G: {obs_com[1]: .2f} | B: {obs_com[2]: .2f}\n'
            f'RGB rcvr. Right | R: {obs_com[3]: .2f} | G: {obs_com[4]: .2f} | B: {obs_com[5]: .2f}\n'
            f'RGB rcvr. Back  | R: {obs_com[6]: .2f} | G: {obs_com[7]: .2f} | B: {obs_com[8]: .2f}\n'
            f'RGB rcvr. Left  | R: {obs_com[9]: .2f} | G: {obs_com[10]: .2f} | B: {obs_com[11]: .2f}\n')

    def get_rendered_images(self) -> 'tuple[np.ndarray, np.ndarray, np.ndarray]':
        self.gym.render_all_camera_sensors(self.sim_handle)
        self.gym.start_access_image_tensors(self.sim_handle)

        rgb = self.session.img_rgb_list[self.all_bot_idx][..., :3]
        dep = self.session.img_dep_list[self.all_bot_idx]
        typ = self.session.img_seg_list[self.all_bot_idx]

        if self.sim.is_preset:
            typ_mask = (typ == cfg.SEG_CLS_NULL).unsqueeze(-1).float()
            rgb = torch.lerp(rgb.float(), self.session.sky_clr, typ_mask)

        rgb = rgb.cpu().numpy()
        dep = 255. * norm_depth_range(-dep, MAX_IMG_DEPTH).cpu().numpy()
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
                    # self.update_top_view()

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
                        self.session.actions[:, -cfg.RGB_VEC_SIZE:] = self.session.sample_colours()

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

    def update_preview(self):
        if not self.session.preview:
            return

        step_out = self.session.async_temp_result

        if step_out is None or len(step_out) != 4 or len(step_out[0].shape) != 4:
            return

        rgb, dep, typ = (self.get_visnet_images if self.prev_reconstruct else self.get_rendered_images)()
        stacked = np.concatenate((rgb, np.stack((dep,)*3, axis=-1), np.stack((typ,)*3, axis=-1)), axis=0)

        data = f'P6 {self.PREV_DIM[0]} {self.PREV_DIM[1]} 255 '.encode() + stacked.astype(np.uint8).tobytes()

        photoimage = tk.PhotoImage(
            width=self.PREV_DIM[0],
            height=self.PREV_DIM[1],
            data=data,
            format='PPM'
        ).zoom(self.PREV_ZOOM, self.PREV_ZOOM)

        self.tk_canvas.create_image(0, 0, anchor='nw', image=photoimage)
        self.tk_root.update()

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

    REC_ALL = 3
    REC_IMG = 2
    REC_VEC = 1
    REC_NONE = 0

    ARGS = [
        {'name': '--level', 'type': int, 'default': 4, 'help': 'Maze complexity level.'},
        {'name': '--preset_name', 'type': str, 'default': 'env7', 'help': 'Name of files with preset level data.'},
        {'name': '--regen', 'type': int, 'default': 0, 'help': 'Option to fully regenerate environments on reset.'},
        {'name': '--n_bots', 'type': int, 'default': -1, 'help': 'Number of agents per environment.'},
        {'name': '--n_envs', 'type': int, 'default': -1, 'help': 'Number of parallel environments.'},
        {'name': '--x_duration', 'type': int, 'default': 1, 'help': 'Episode duration multiplier.'},
        {'name': '--end_step', 'type': int, 'default': -1, 'help': 'Max steps until auto-termination.'},
        {'name': '--ctrl_mode', 'type': int, 'default': CTRL_MAN, 'help': 'Sim/agent control mode.'},
        {'name': '--rec_mode', 'type': int, 'default': REC_NONE, 'help': 'Data category to record.'},
        {'name': '--preview', 'type': int, 'default': 0, 'help': 'Option to view input or recons. images in side GUI.'},
        {'name': '--headless', 'type': int, 'default': 0, 'help': 'Option to run without a viewer.'},
        {'name': '--act_freq', 'type': int, 'default': cfg.STEPS_PER_SECOND, 'help': 'Inference steps per second.'},
        {'name': '--transfer_name', 'type': str, 'default': '', 'help': 'Starting model name/ID string.'},
        {'name': '--transfer_ver', 'type': int, 'default': 0, 'help': 'Starting model ckpt. version.'},
        {'name': '--model_name', 'type': str, 'default': 'mazeai', 'help': 'Model name/ID string.'},
        {'name': '--com_level', 'type': int, 'default': Policy.COM_FREE, 'help': 'Level of communication.'},
        {'name': '--guide_level', 'type': int, 'default': Policy.GUIDE_FREE, 'help': 'Level of objective guidance.'},
        {'name': '--aux_level', 'type': int, 'default': Policy.AUX_FREE, 'help': 'Level of aux. objective alignment.'},
        {'name': '--prob_actor', 'type': int, 'default': 1, 'help': 'Option to keep probabilistic inference.'},
        {'name': '--rew_sharing', 'type': int, 'default': 0, 'help': 'Option to enable reward sharing.'},
        {'name': '--rng_seed', 'type': int, 'default': 42, 'help': 'Seed for numpy and torch RNGs.'}]

    def __init__(self, args: Namespace):
        self.end_step: int = args.end_step
        self.ctrl_mode: int = args.ctrl_mode
        self.rec_mode: int = args.rec_mode
        self.rec_data_queue: 'list[Tensor]' = []
        self.preview = bool(args.preview)

        # Resume model state
        self.model_options = {
            'com_level': args.com_level,
            'guide_level': args.guide_level,
            'aux_level': args.aux_level,
            'prob_actor': args.prob_actor}

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
        sim.ep_duration: int = round(sim.ep_duration * args.x_duration)

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
            reward_sharing=bool(args.rew_sharing),
            device=args.sim_device)

        self.accelerate()

    def post_step(
        self,
        obs: 'tuple[Tensor, ...]',
        reward: Tensor,
        rst_mask_f: Tensor,
        _vals: 'tuple[Tensor, ...]' = None,
        _info: 'dict[str, Any]' = None
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

        img, vec, com = obs

        # Images are stored in full or vector form (as means)
        if self.rec_mode == self.REC_VEC:
            img = img.mean((-2, -1))

        vecs = (vec, rew.unsqueeze(-1), rst.unsqueeze(-1))

        if self.rec_mode < self.REC_ALL:
            com = Policy.weigh_signals(self.sim, vec[:, cfg.OBS_RGB_SLICE], com)

        img_data = img.cpu().numpy()
        vec_data = torch.hstack(vecs).cpu().numpy()
        com_data = com.cpu().numpy()

        self.rec_data_queue.extend((img_data, vec_data, com_data))

    def save_rec_data(self):
        if self.rec_mode == self.REC_NONE or not self.rec_data_queue:
            return

        file_idx = get_available_file_idx(cfg.DATA_DIR, 'rec')
        filename = os.path.join(cfg.DATA_DIR, f'rec_{file_idx:02d}.npz')

        img = np.stack(self.rec_data_queue[0::3])
        vec = np.stack(self.rec_data_queue[1::3])
        com = np.stack(self.rec_data_queue[2::3])
        self.rec_data_queue.clear()

        np.savez_compressed(filename, img=img, vec=vec, com=com)

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

        optimiser = NAdamW(
            (param for param in model.parameters() if param.requires_grad),
            lr=get_arg_defaults(AdaptivePlateauScheduler.__init__)['lr_milestones'][0],
            weight_decay=cfg.WEIGHT_DECAY)

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

        n_batches_per_step = self.sim.n_all_bots // MazeSim.NUM_ALL_BOTS - MazeSim.NUM_ALL_BOTS // self.sim.n_all_bots
        n_rollouts = round(cfg.N_ROLLOUT_STEPS * MazeSim.NUM_ALL_BOTS / self.sim.n_all_bots)

        rl_algo = PPG(
            self.step,
            self.ckpter,
            scheduler,
            cfg.N_EPOCHS_MAP[self.sim.level],
            cfg.LOG_EPOCH_INTERVAL,
            cfg.CKPT_EPOCH_INTERVAL,
            cfg.BRANCH_EPOCH_INTERVAL,
            n_rollouts,
            cfg.N_TRUNCATED_STEPS,
            self.sim.n_all_bots,
            n_batches_per_step,
            cfg.N_ROLLOUTS_PER_EPOCH,
            cfg.N_AUX_ITERS_PER_EPOCH,
            gamma,
            aux_weight=cfg.AUX_WEIGHT,
            entropy_weight=cfg.ENT_WEIGHT,
            log_dir=cfg.LOG_DIR)

        try:
            rl_algo.run()

        except KeyboardInterrupt:
            rl_algo.writer.close()
            raise

    def eval(self):
        if not self.headless:
            self.interface.update_top_view()

        model = ActorCritic(self.sim.n_bots, self.sim.n_envs, **self.model_options)

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

                obs, reward, rst_mask_f, _, _ = self.step(actions, get_info=False)
                obs = self.post_step(obs, reward, rst_mask_f)

                if rst_mask_f.any().item():
                    mem = model.reset_mem(mem, rst_mask_f)

    def play(self):
        if not self.headless:
            self.interface.update_top_view()

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
