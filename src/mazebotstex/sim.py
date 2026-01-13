"""Gym wrappers"""

import os
from argparse import Namespace

import numpy as np
from numpy import ndarray
from scipy.spatial.transform import Rotation
from isaacgym import gymapi

import config as cfg
from maze import MazeConstructor
from utils import get_cached_paths, get_numba_dict, read_texture_file, warp_basic, warp_shear


# ------------------------------------------------------------------------------
# MARK: MazeEnv

class MazeEnv:
    """`Gym.Env` wrapper for env. setup and path estimation."""

    GOAL_CLRS = np.array([gymapi.Vec3(*clr) for clr in cfg.COLOURS['goal']], dtype=object)
    WALL_CLRS = np.array([gymapi.Vec3(*clr) for clr in cfg.COLOURS['wall']], dtype=object)
    NEUTRAL_CLRS = np.array([gymapi.Vec3(*clr) for clr in cfg.COLOURS['neutral']], dtype=object)
    FLOOR_CLR = gymapi.Vec3(*cfg.COLOURS['neutral'][cfg.FLOOR_CLR_IDX])

    MAX_SIDE_LENGTH = cfg.LEVEL_PARAMS[max(cfg.LEVEL_PARAMS)]['side_length']
    FLOOR_POSE = (0., 0., -cfg.WALL_WIDTH / 2.)

    wall_handles: ndarray
    blk_handles: ndarray
    link_handles: ndarray
    decoy_handles: ndarray
    obj_handles: ndarray
    bot_handles: ndarray
    cam_handles: ndarray

    wall_idcs: ndarray
    blk_idcs: ndarray
    link_idcs: ndarray
    decoy_idcs: ndarray
    obj_idcs: ndarray
    bot_idcs: ndarray

    data: 'dict[str, ndarray]'
    cell_pass_map: 'dict[int, ndarray]'

    def __init__(self, sim: 'MazeSim', sampler: MazeConstructor, idx: int, bot_idx: int, text_mode: int):
        self.sim = sim
        self.sampler = sampler
        self.idx = idx
        self.bot_idx = bot_idx
        self.bot_slice = slice(bot_idx, bot_idx + sampler.n_bots)
        self.text_mode = text_mode

        # Place environment onto the grid
        bbox_side_halflength = self.MAX_SIDE_LENGTH / 2. + cfg.ENV_HALFSPACING

        bbox_vertex_low = gymapi.Vec3(-bbox_side_halflength, -bbox_side_halflength, 0.)
        bbox_vertex_high = gymapi.Vec3(bbox_side_halflength, bbox_side_halflength, cfg.WALL_HEIGHT)

        self.handle = sim.gym.create_env(
            sim.handle,
            bbox_vertex_low,
            bbox_vertex_high,
            sim.n_envs_per_row)

        self.delim_pts = sampler.delim_pt_grid.reshape(-1, 2)
        self.wall_pt_tuple = (sampler.wall_w_grid, sampler.wall_n_grid)
        self.wall_pts = np.concatenate([wall_pts.reshape(-1, 2) for wall_pts in self.wall_pt_tuple])
        wall_angles = np.concatenate(([0.]*(len(self.wall_pts)//2), [np.pi/2.]*(len(self.wall_pts)//2)))
        wall_quats = Rotation.from_euler('z', wall_angles).as_quat()

        # Place entities into the environment
        self.resample()
        self.cellpair_path_map: 'dict[tuple[int, int], ndarray]' = get_numba_dict(tuple_as_key=True)
        self.create_static()
        self.create_bots()
        self.set_dof_props()
        self.set_rigid_props()
        self.recolour()

        # Prep. for path estimation
        self.cell_pts = sampler.cell_pt_grid.reshape(-1, 2)
        self.open_delims = sampler.open_delims

        # Data for env. resetting
        n_links = (sampler.n_side_divs + 1) ** 2
        n_cells = sampler.n_side_divs ** 2

        self.wall_substate = np.concatenate((
            wall_quats,
            np.zeros((len(wall_quats), 6), dtype=np.float32)), axis=-1, dtype=np.float32)

        self.link_substate = np.concatenate((
            np.zeros((n_links, 3), dtype=np.float32),
            np.ones((n_links, 1), dtype=np.float32),
            np.zeros((n_links, 6), dtype=np.float32)), axis=-1, dtype=np.float32)

        self.blk_substate = self.link_substate[:n_cells]

        self.obj_substate = np.concatenate((
            np.ones((sampler.n_goals, 1), dtype=np.float32) * cfg.OBJ_HEIGHT,
            np.zeros((sampler.n_goals, 3), dtype=np.float32),
            np.ones((sampler.n_goals, 1), dtype=np.float32),
            np.zeros((sampler.n_goals, 6), dtype=np.float32)), axis=-1, dtype=np.float32)

        self.decoy_substate = np.concatenate((
            np.ones((sampler.n_decoys, 1), dtype=np.float32) * cfg.OBJ_HEIGHT,
            np.zeros((sampler.n_decoys, 3), dtype=np.float32),
            np.ones((sampler.n_decoys, 1), dtype=np.float32),
            np.zeros((sampler.n_decoys, 6), dtype=np.float32)), axis=-1, dtype=np.float32)

        self.bot_substates = (
            np.zeros((sampler.n_bots, 1), dtype=np.float32),
            np.zeros((sampler.n_bots, 6), dtype=np.float32))

    # --------------------------------------------------------------------------
    # MARK: resample

    def resample(self):
        """Reinit. the path map, points, and goal distribution."""

        self.data = self.sampler.generate()
        self.cell_wall_grid = self.data.cell_wall_grid

        # Convert passage map into numba typed dict.
        self.cell_pass_map = get_numba_dict()

        for key, lst in self.data.cell_pass_map.items():
            self.cell_pass_map[key] = np.array(lst, dtype=np.int64)

    # --------------------------------------------------------------------------
    # MARK: create_static

    def create_static(self):
        """Place objects at target locations and raise or lower structural blocks."""

        gym = self.sim.gym

        link_mask = self.data.link_mask
        block_mask = self.data.block_mask

        wall_mask = np.concatenate((
            self.data.wall_mask[:self.sampler.n_side_divs, :, cfg.SIDE_W_IDX].ravel(),
            self.data.wall_mask[:, :self.sampler.n_side_divs, cfg.SIDE_N_IDX].ravel()))

        n_walls = len(self.wall_pts) // 2

        self.base_handles = np.zeros(1, dtype=np.int32)
        self.wall_handles = np.zeros(wall_mask.shape, dtype=np.int32)
        self.blk_handles = np.zeros(block_mask.shape, dtype=np.int32)
        self.link_handles = np.zeros(link_mask.shape, dtype=np.int32)
        self.decoy_handles = np.zeros(self.sampler.n_decoys, dtype=np.int32)
        self.obj_handles = np.zeros(self.sampler.n_goals, dtype=np.int32)

        self.base_idcs = np.zeros(1, dtype=np.int32)
        self.wall_idcs = np.zeros(wall_mask.shape, dtype=np.int32)
        self.blk_idcs = np.zeros(block_mask.shape, dtype=np.int32)
        self.link_idcs = np.zeros(link_mask.shape, dtype=np.int32)
        self.decoy_idcs = np.zeros(self.sampler.n_decoys, dtype=np.int32)
        self.obj_idcs = np.zeros(self.sampler.n_goals, dtype=np.int32)

        self.base_handles[0] = floor_handle = gym.create_actor(
            self.handle,
            self.sim.assets['floor'],
            gymapi.Transform(gymapi.Vec3(*self.FLOOR_POSE)),
            'floor',
            self.idx,
            -1,
            cfg.ENT_CLS_PLANE)

        self.base_idcs[0] = gym.get_actor_index(self.handle, floor_handle, gymapi.DOMAIN_SIM)

        wallast = self.sim.assets['wall' if self.sampler.level == 1 else 'wall2']
        roofast = self.sim.assets['block' if self.sampler.level == 1 else 'block2']

        for i, pos in enumerate(self.wall_pts):
            pose = gymapi.Transform(gymapi.Vec3(
                *pos,
                cfg.BLOCK_HALFHEIGHT if wall_mask[i] else cfg.BLOCK_HIDDENHEIGHT),
                None if i < n_walls else gymapi.Quat.from_euler_zyx(0., 0., np.pi/2.))

            self.wall_handles[i] = wall_handle = gym.create_actor(
                self.handle,
                wallast,
                pose,
                f'wall-{i:02d}',
                self.idx,
                -1,
                cfg.ENT_CLS_WALL)

            self.wall_idcs[i] = gym.get_actor_index(self.handle, wall_handle, gymapi.DOMAIN_SIM)

        for i in range(self.sampler.n_side_divs):
            for j in range(self.sampler.n_side_divs):
                pose = gymapi.Transform(gymapi.Vec3(
                    *self.sampler.cell_pt_grid[i, j],
                    cfg.BLOCK_HALFHEIGHT if block_mask[i, j] else cfg.BLOCK_HIDDENHEIGHT))

                self.blk_handles[i, j] = blk_handle = gym.create_actor(
                    self.handle,
                    roofast,
                    pose,
                    f'block-{i:02d}-{j:02d}',
                    self.idx,
                    -1,
                    cfg.ENT_CLS_WALL)

                self.blk_idcs[i, j] = gym.get_actor_index(self.handle, blk_handle, gymapi.DOMAIN_SIM)

        self.blk_idcs = self.blk_idcs.flatten()

        for i in range(self.sampler.n_side_divs + 1):
            for j in range(self.sampler.n_side_divs + 1):
                pose = gymapi.Transform(gymapi.Vec3(
                    *self.sampler.delim_pt_grid[i, j],
                    cfg.BLOCK_HALFHEIGHT if link_mask[i, j] else cfg.BLOCK_HIDDENHEIGHT))

                self.link_handles[i, j] = link_handle = gym.create_actor(
                    self.handle,
                    self.sim.assets['link'],
                    pose,
                    f'link-{i:02d}-{j:02d}',
                    self.idx,
                    -1,
                    cfg.ENT_CLS_WALL)

                self.link_idcs[i, j] = gym.get_actor_index(self.handle, link_handle, gymapi.DOMAIN_SIM)

        self.link_idcs = self.link_idcs.flatten()

        for i, pos in enumerate(self.data.decoy_pts):
            pose = gymapi.Transform(gymapi.Vec3(*pos, cfg.OBJ_HEIGHT))

            self.decoy_handles[i] = decoy_handle = gym.create_actor(
                self.handle,
                self.sim.assets['obj'],
                pose,
                f'decoy-{i:d}',
                self.idx,
                -1,
                cfg.ENT_CLS_CLUTTER)

            self.decoy_idcs[i] = gym.get_actor_index(self.handle, decoy_handle, gymapi.DOMAIN_SIM)

        for i, pos in enumerate(self.data.obj_pts):
            pose = gymapi.Transform(gymapi.Vec3(*pos, cfg.OBJ_HEIGHT))

            self.obj_handles[i] = obj_handle = gym.create_actor(
                self.handle,
                self.sim.assets['obj'],
                pose,
                f'goal-{i:d}',
                self.idx,
                -1,
                cfg.ENT_CLS_OBJ)

            self.obj_idcs[i] = gym.get_actor_index(self.handle, obj_handle, gymapi.DOMAIN_SIM)

    # --------------------------------------------------------------------------
    # MARK: create_bots

    def create_bots(self):
        """Place agents at spawn points and attach camera sensors."""

        gym = self.sim.gym

        self.bot_handles = np.zeros(self.sampler.n_bots, dtype=np.int32)
        self.bot_idcs = np.zeros(self.sampler.n_bots, dtype=np.int32)

        for i, pos, angle in zip(range(self.sampler.n_bots), self.data.spawn_pts, self.data.spawn_angles):
            pose = gymapi.Transform(
                gymapi.Vec3(*pos, 0.),
                gymapi.Quat.from_euler_zyx(0., 0., angle))

            self.bot_handles[i] = bot_handle = gym.create_actor(
                    self.handle,
                    self.sim.assets['bot'],
                    pose,
                    f'bot-{i:03d}',
                    self.idx,
                    -1,
                    cfg.ENT_CLS_BOT)

            self.bot_idcs[i] = gym.get_actor_index(self.handle, bot_handle, gymapi.DOMAIN_SIM)

        # Init. visual sensors
        camera_props = gymapi.CameraProperties()
        camera_props.enable_tensors = True
        camera_props.height = cfg.OBS_IMG_RES_HEIGHT
        camera_props.width = cfg.OBS_IMG_RES_WIDTH
        camera_props.horizontal_fov = 55.
        camera_props.far_plane = 64.
        camera_pos = gymapi.Vec3(*cfg.CAM_OFFSET)
        camera_ori = gymapi.Quat.from_euler_zyx(0., np.pi/16-0.016, 0.)

        self.cam_handles = np.zeros_like(self.bot_handles)

        for i, bot_handle in enumerate(self.bot_handles):
            self.cam_handles[i] = cam_handle = gym.create_camera_sensor(self.handle, camera_props)
            body_handle = gym.get_actor_rigid_body_handle(self.handle, bot_handle, cfg.BOT_BODY_IDX)

            gym.attach_camera_to_body(
                cam_handle, self.handle, body_handle, gymapi.Transform(camera_pos, camera_ori), gymapi.FOLLOW_TRANSFORM)

    # --------------------------------------------------------------------------
    # MARK: recolour

    def recolour(self):
        if self.text_mode != cfg.TEXT_NONE:
            return self.retexture()

        set_clr = self.sim.gym.set_rigid_body_color
        set_sid = self.sim.gym.set_rigid_body_segmentation_id

        # Objects
        obj_clrs = self.GOAL_CLRS[self.data.obj_goal_map]

        for obj_handle, clr, clr_idx in zip(self.obj_handles, obj_clrs, self.data.obj_goal_map):
            set_clr(self.handle, obj_handle, 0, gymapi.MESH_VISUAL, clr)
            set_sid(self.handle, obj_handle, 0, cfg.OBJ_CLS_OFFSET + clr_idx)

        # Clutter
        for decoy_handle in self.decoy_handles:
            set_clr(self.handle, decoy_handle, 0, gymapi.MESH_VISUAL, self.NEUTRAL_CLRS[cfg.WHITE_CLR_IDX])

        # Bots
        for bot_handle in self.bot_handles:
            clr = self.NEUTRAL_CLRS[cfg.BOT_CLR_IDX]
            set_clr(self.handle, bot_handle, cfg.BOT_BODY_IDX, gymapi.MESH_VISUAL, clr)
            set_clr(self.handle, bot_handle, cfg.BOT_POLE_IDX, gymapi.MESH_VISUAL, self.NEUTRAL_CLRS[cfg.WHEEL_CLR_IDX])
            set_clr(self.handle, bot_handle, cfg.BOT_CAM_IDX, gymapi.MESH_VISUAL, clr)
            set_clr(self.handle, bot_handle, cfg.BOT_CHAS_IDX, gymapi.MESH_VISUAL, clr)

            whlclr = self.NEUTRAL_CLRS[cfg.WHEEL_CLR_IDX]

            for i in range(cfg.N_BOT_WHLS):
                set_clr(self.handle, bot_handle, cfg.BOT_WHL1_IDX+i, gymapi.MESH_VISUAL, whlclr)

        # Base
        set_clr(self.handle, self.base_handles[0], 0, gymapi.MESH_VISUAL, self.NEUTRAL_CLRS[cfg.FLOOR_CLR_IDX])

        for i, link_handle in enumerate(self.link_handles.ravel()):
            clr_idx = max(self.data.link_clr_idcs[i], 0)
            set_clr(self.handle, link_handle, 0, gymapi.MESH_VISUAL, self.WALL_CLRS[clr_idx])

        # Walls and blocks
        for i, wall_handle in enumerate(self.wall_handles):
            clr_idx = max(self.data.wall_clr_idcs[i], 0)
            set_clr(self.handle, wall_handle, 0, gymapi.MESH_VISUAL, self.WALL_CLRS[clr_idx])

        for i in range(self.blk_handles.shape[0]):
            for j in range(self.blk_handles.shape[1]):
                clr_idx = max(self.data.cell_clr_idx_grid[i, j], 0)
                set_clr(self.handle, self.blk_handles[i, j], 0, gymapi.MESH_VISUAL, self.WALL_CLRS[clr_idx])

    # --------------------------------------------------------------------------
    # MARK: retexture

    def retexture(self):
        """Set obj. and bot colours according to associated goals."""

        rng = self.sampler.rng
        set_txt = self.sim.gym.set_rigid_body_texture
        set_clr = self.sim.gym.set_rigid_body_color
        set_sid = self.sim.gym.set_rigid_body_segmentation_id

        use_same = (self.text_mode >= cfg.TEXT_RANDAUG) and (rng.random() > 0.5)

        if use_same:
            tset = {}

            for k, v in self.sim.textures.items():
                tset[k] = [rng.choice(v)] if v else []

        else:
            tset = self.sim.textures

        # Objects
        obj_clrs = self.GOAL_CLRS[self.data.obj_goal_map]

        for obj_handle, clr, clr_idx in zip(self.obj_handles, obj_clrs, self.data.obj_goal_map):
            set_txt(self.handle, obj_handle, 0, gymapi.MESH_VISUAL, rng.choice(tset['obj']))
            set_clr(self.handle, obj_handle, 0, gymapi.MESH_VISUAL, clr)
            set_sid(self.handle, obj_handle, 0, cfg.OBJ_CLS_OFFSET + clr_idx)

        # Clutter
        for decoy_handle in self.decoy_handles:
            set_txt(self.handle, decoy_handle, 0, gymapi.MESH_VISUAL, rng.choice(tset['dec']))

        # Bots
        for bot_handle in self.bot_handles:
            set_txt(self.handle, bot_handle, cfg.BOT_BODY_IDX, gymapi.MESH_VISUAL, rng.choice(tset['bot']))
            set_txt(self.handle, bot_handle, cfg.BOT_POLE_IDX, gymapi.MESH_VISUAL, rng.choice(tset['pld']))
            set_txt(self.handle, bot_handle, cfg.BOT_CAM_IDX, gymapi.MESH_VISUAL, rng.choice(tset['cam']))
            set_txt(self.handle, bot_handle, cfg.BOT_CHAS_IDX, gymapi.MESH_VISUAL, rng.choice(tset['chs']))

            whltext = rng.choice(tset['whl'])

            for i in range(4):
                set_txt(self.handle, bot_handle, cfg.BOT_WHL1_IDX+i, gymapi.MESH_VISUAL, whltext)

        # Base
        set_txt(self.handle, self.base_handles[0], 0, gymapi.MESH_VISUAL, rng.choice(tset['flr']))

        n_side_divs = self.sampler.n_side_divs
        n_walls_per_side = (n_side_divs + 1) * n_side_divs

        wall_w_handles = self.wall_handles[:n_walls_per_side].reshape(self.wall_pt_tuple[0].shape[:2])
        wall_n_handles = self.wall_handles[n_walls_per_side:].reshape(self.wall_pt_tuple[1].shape[:2])

        wall_w_clrs = self.data.wall_clr_idcs[:n_walls_per_side].reshape(self.wall_pt_tuple[0].shape[:2])
        wall_n_clrs = self.data.wall_clr_idcs[n_walls_per_side:].reshape(self.wall_pt_tuple[1].shape[:2])

        wallclrs = rng.choice(tset['blk'], cfg.N_WALL_CLRS)
        walltext = rng.choice(wallclrs)

        if not use_same and tset['brd']:
            brdtext = rng.choice(tset['brd'])

            for i in range(self.link_handles.shape[0]):
                set_txt(self.handle, self.link_handles[i, 0], 0, gymapi.MESH_VISUAL, brdtext)
                set_txt(self.handle, self.link_handles[i, -1], 0, gymapi.MESH_VISUAL, brdtext)

            for j in range(self.link_handles.shape[1]):
                set_txt(self.handle, self.link_handles[0, j], 0, gymapi.MESH_VISUAL, brdtext)
                set_txt(self.handle, self.link_handles[-1, j], 0, gymapi.MESH_VISUAL, brdtext)

            if self.text_mode == cfg.TEXT_WARES:
                link_clr_idx_grid = self.data.link_clr_idcs.reshape(self.link_handles.shape)

                for i in range(1, self.link_handles.shape[0] - 1):
                    for j in range(1, self.link_handles.shape[1] - 1):
                        clr_idx = max(link_clr_idx_grid[i, j], 0)
                        set_txt(self.handle, self.link_handles[i, j], 0, gymapi.MESH_VISUAL, wallclrs[clr_idx])

            for i in range(wall_w_handles.shape[0]):
                set_txt(self.handle, wall_w_handles[i, 0], 0, gymapi.MESH_VISUAL, brdtext)
                set_txt(self.handle, wall_w_handles[i, -1], 0, gymapi.MESH_VISUAL, brdtext)

            for j in range(wall_n_handles.shape[1]):
                set_txt(self.handle, wall_n_handles[0, j], 0, gymapi.MESH_VISUAL, brdtext)
                set_txt(self.handle, wall_n_handles[-1, j], 0, gymapi.MESH_VISUAL, brdtext)

            for i in range(wall_w_handles.shape[0]):
                for j in range(1, wall_w_handles.shape[1] - 1):
                    clr_idx = max(wall_w_clrs[i, j], 0)
                    set_txt(self.handle, wall_w_handles[i, j], 0, gymapi.MESH_VISUAL, wallclrs[clr_idx])

            for j in range(wall_n_handles.shape[1]):
                for i in range(1, wall_n_handles.shape[0] - 1):
                    clr_idx = max(wall_n_clrs[i, j], 0)
                    set_txt(self.handle, wall_n_handles[i, j], 0, gymapi.MESH_VISUAL, wallclrs[clr_idx])

        elif self.text_mode == cfg.TEXT_WARES:
            for i, link_handle in enumerate(self.link_handles.ravel()):
                clr_idx = max(self.data.link_clr_idcs[i], 0)
                set_clr(self.handle, link_handle, 0, gymapi.MESH_VISUAL, wallclrs[clr_idx])

        else:
            linktext = walltext if use_same else rng.choice(tset['lnk'])

            for link_handle in self.link_handles.ravel():
                set_txt(self.handle, link_handle, 0, gymapi.MESH_VISUAL, linktext)

        # Walls
        if use_same:
            for i, wall_handle in enumerate(self.wall_handles):
                set_txt(self.handle, wall_handle, 0, gymapi.MESH_VISUAL, walltext)

        elif not tset['brd']:
            for i, wall_handle in enumerate(self.wall_handles):
                clr_idx = max(self.data.wall_clr_idcs[i], 0)
                set_txt(self.handle, wall_handle, 0, gymapi.MESH_VISUAL, wallclrs[clr_idx])

        # Blocks
        if use_same:
            for blk_handle in self.blk_handles.ravel():
                set_txt(self.handle, blk_handle, 0, gymapi.MESH_VISUAL, walltext)

        else:
            for i in range(self.blk_handles.shape[0]):
                for j in range(self.blk_handles.shape[1]):
                    clr_idx = max(self.data.cell_clr_idx_grid[i, j], 0)
                    set_txt(self.handle, self.blk_handles[i, j], 0, gymapi.MESH_VISUAL, wallclrs[clr_idx])

    # --------------------------------------------------------------------------
    # MARK: set_dof_props

    def set_dof_props(self):
        """Set bot motor properties."""

        gym = self.sim.gym

        mul_min = 1. - cfg.MAX_PARAM_OFFSET
        mul_range = 2 * cfg.MAX_PARAM_OFFSET

        stiffnesses = cfg.MOT_STIFFNESS * (mul_min + mul_range * self.sampler.rng.random(self.sampler.n_bots))
        dampings = cfg.MOT_DAMPING * (mul_min + mul_range * self.sampler.rng.random(self.sampler.n_bots))

        dof_props: 'dict[str, ndarray]' = gym.get_actor_dof_properties(self.handle, self.bot_handles[0])
        dof_props['driveMode'].fill(gymapi.DOF_MODE_EFFORT)

        for bot_handle, stiffness, damping in zip(self.bot_handles, stiffnesses, dampings):
            dof_props['stiffness'].fill(stiffness)
            dof_props['damping'].fill(damping)

            gym.set_actor_dof_properties(self.handle, bot_handle, dof_props)

    # --------------------------------------------------------------------------
    # MARK: set_rigid_props

    def set_rigid_props(self):
        """
        Set bot interaction properties.

        NOTE: Calling `set_actor_rigid_shape_properties` more than once results
        in unpredictable behaviour, like bots being able to pass through some
        moved (reset) blocks (though still colliding between themselves).
        """

        gym = self.sim.gym

        mul_min = 1. - cfg.MAX_PARAM_OFFSET
        mul_range = 2 * cfg.MAX_PARAM_OFFSET

        frictions = cfg.WHL_FRICTION * (mul_min + mul_range * self.sampler.rng.random(self.sampler.n_bots))

        for bot_handle, friction in zip(self.bot_handles, frictions):
            shape_prop_list = gym.get_actor_rigid_shape_properties(self.handle, bot_handle)

            for shape_props in shape_prop_list:
                shape_props.friction = friction

            gym.set_actor_rigid_shape_properties(self.handle, bot_handle, shape_prop_list)

    # --------------------------------------------------------------------------
    # MARK: reset

    def reset(self, full: bool = True) -> ndarray:
        """Partially or fully reset the environment and relay the new states."""

        if full or self.data.global_spawn_flag:
            self.resample()
            self.recolour()
            self.set_dof_props()

        else:
            self.sampler.refresh(self.data)

        self.cellpair_path_map = get_numba_dict(tuple_as_key=True)

        wall_states = np.concatenate((
            self.wall_pts,
            np.where(
                np.concatenate((
                    self.data.wall_mask[:self.sampler.n_side_divs, :, cfg.SIDE_W_IDX].ravel(),
                    self.data.wall_mask[:, :self.sampler.n_side_divs, cfg.SIDE_N_IDX].ravel())),
                cfg.BLOCK_HALFHEIGHT,
                cfg.BLOCK_HIDDENHEIGHT).astype(np.float32)[:, None],
            self.wall_substate), axis=-1, dtype=np.float32)

        blk_states = np.concatenate((
            self.cell_pts,
            np.where(
                self.data.block_mask.ravel(),
                cfg.BLOCK_HALFHEIGHT,
                cfg.BLOCK_HIDDENHEIGHT).astype(np.float32)[:, None],
            self.blk_substate), axis=-1, dtype=np.float32)

        link_states = np.concatenate((
            self.delim_pts,
            np.where(
                self.data.link_mask.ravel(),
                cfg.BLOCK_HALFHEIGHT,
                cfg.BLOCK_HIDDENHEIGHT).astype(np.float32)[:, None],
            self.link_substate), axis=-1, dtype=np.float32)

        obj_states = np.concatenate((
            self.data.obj_pts,
            self.obj_substate), axis=-1, dtype=np.float32)

        decoy_states = np.concatenate((
            self.data.decoy_pts,
            self.decoy_substate), axis=-1, dtype=np.float32)

        bot_states = np.concatenate((
            self.data.spawn_pts,
            self.bot_substates[0],
            Rotation.from_euler('z', self.data.spawn_angles).as_quat(),
            self.bot_substates[1]), axis=-1, dtype=np.float32)

        actor_states = np.concatenate((wall_states, blk_states, link_states, obj_states, decoy_states, bot_states))

        return actor_states

    # --------------------------------------------------------------------------
    # MARK: get_path_estimates

    def get_path_estimates(
        self,
        start_pts: ndarray,
        end_pts: ndarray,
        sight_mask: ndarray,
    ) -> 'tuple[ndarray, ndarray]':
        """
        Use A* on the underlying graph to estimate path length and direction
        from valid starting points to target objects.
        """

        return get_cached_paths(
            start_pts,
            end_pts,
            sight_mask,
            self.cell_pass_map,
            self.cellpair_path_map,
            self.open_delims,
            self.cell_pts,
            self.cell_wall_grid)


# ------------------------------------------------------------------------------
# MARK: MazeSim

class MazeSim:
    """`Gym.Sim` wrapper for general setup and env. handling."""

    MAX_FPS_PHYSX = 128

    DEFAULT_SIM_ARGS = Namespace(
        global_spawn_prob=0.,
        draw_freq=64,
        use_gpu_pipeline=True,
        use_gpu=True,
        num_threads=0,
        compute_device_id=0,
        graphics_device_id=0)

    LIGHT_DIRECTIONS = (
        gymapi.Vec3(1., 1., 2.),
        gymapi.Vec3(1., -1., 2.),
        gymapi.Vec3(-1., 0.33, 2.),
        gymapi.Vec3(-1., -0.33, 2.))

    def __init__(self, args: Namespace = DEFAULT_SIM_ARGS, rng: 'None | int | np.random.Generator' = None):

        # Init. Isaac Gym
        self.gym = gym = gymapi.acquire_gym()

        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0., 0., -cfg.GRAV_CONST)
        sim_params.dt = 1. / args.draw_freq
        sim_params.substeps = max(self.MAX_FPS_PHYSX // args.draw_freq, 1)

        # NOTE: Params. behave differently between CPU and GPU
        sim_params.use_gpu_pipeline = args.use_gpu_pipeline     # Sim on GPU with tensor wrapping
        sim_params.physx.use_gpu = args.use_gpu                 # Sim on GPU
        sim_params.physx.num_threads = args.num_threads         # n_cpu_cores-1 by default
        sim_params.physx.num_position_iterations = 3            # 4/6/8 in gym examples, but 2 works as well
        sim_params.physx.num_velocity_iterations = 3            # 0/1 in gym examples, but higher is more stable here
        sim_params.physx.max_depenetration_velocity = 7.5       # 100. default seemed a bit too violent
        sim_params.physx.rest_offset = 0.                       # 0.001 by default
        sim_params.physx.contact_offset = 0.005                 # 0.02 default is relatively large wrt. bot dimensions

        # All substeps by default, but that produces noisy and nondeterministic results
        sim_params.physx.contact_collection = gymapi.CC_LAST_SUBSTEP

        # Assert device consistency
        if args.use_gpu_pipeline and (args.graphics_device_id != args.compute_device_id):
            print(f'Warning: Overriding graphics device {args.graphics_device_id} to {args.compute_device_id}.')

            args.graphics_device_id = args.compute_device_id

        # Init. PhysX
        self.handle = gym.create_sim(args.compute_device_id, args.graphics_device_id, gymapi.SIM_PHYSX, sim_params)

        if self.handle is None:
            raise Exception('Failed to initialise.')

        # Override lighting to normalise viewing conditions
        self.relight()

        # Load assets
        bot_asset_options = gymapi.AssetOptions()
        bot_asset_options.thickness = 0.005  # 0.02 default; change not visible, but this relates better to bot size

        static_asset_options = gymapi.AssetOptions()
        static_asset_options.thickness = 0.005
        static_asset_options.fix_base_link = True

        self.assets = {
            'bot': gym.load_urdf(self.handle, cfg.ASSET_DIR, 'mazebot_v3.urdf', bot_asset_options),
            'obj': gym.load_urdf(self.handle, cfg.ASSET_DIR, 'cube.urdf', static_asset_options),
            'wall': gym.load_asset(self.handle, cfg.ASSET_DIR, 'wall.urdf', static_asset_options),
            'wall2': gym.load_asset(self.handle, cfg.ASSET_DIR, 'wall2.urdf', static_asset_options),
            'link': gym.load_asset(self.handle, cfg.ASSET_DIR, 'link.urdf', static_asset_options),
            'block': gym.load_asset(self.handle, cfg.ASSET_DIR, 'block.urdf', static_asset_options),
            'block2': gym.load_asset(self.handle, cfg.ASSET_DIR, 'block2.urdf', static_asset_options),
            'floor': gym.load_urdf(self.handle, cfg.ASSET_DIR, 'floor.urdf', static_asset_options)}

        # Load textures
        self.text_mode: int = args.text_mode
        self.textures = self.textfill(args.aug_num, rng)

        # Create parallel envs.
        self.n_envs = sum(n_envs for n_envs in cfg.ENV_NUM_LVL_PRESETS[args.env_cfg].values())
        self.n_envs_per_row: int = round(self.n_envs**0.5)

        self.max_level = min(cfg.LEVEL_PARAMS)
        self.level_slices = []
        self.envs = []
        env_idx = 0
        bot_idx = 0
        slc_idx = 0

        # Override to enable split text mode
        if self.text_mode == cfg.TEXT_SPLIT:
            preset = {k: max(1, v//2) for k, v in cfg.ENV_NUM_LVL_PRESETS[args.env_cfg].items()}

            n_split_envs = sum(preset.values())
            self.n_envs = n_split_envs * 2
            self.n_envs_per_row = round(self.n_envs**0.5)

            env_spec_iterator = list(zip(preset.keys(), preset.values(), [self.text_mode]*n_split_envs))
            env_spec_iterator += list(zip(preset.keys(), preset.values(), [cfg.TEXT_NONE]*n_split_envs))

        else:
            preset = cfg.ENV_NUM_LVL_PRESETS[args.env_cfg]
            env_spec_iterator = zip(preset.keys(), preset.values(), [self.text_mode]*self.n_envs)

        for level, n_envs, text_mode in env_spec_iterator:
            sampler = MazeConstructor(
                **cfg.LEVEL_PARAMS[level],
                n_decoys=cfg.LEVEL_PARAMS[level]['n_goals'] * (level if args.clutter_fn == 'prog' else 1),
                global_spawn_prob=args.global_spawn_prob,
                speaker_dropout=args.prob_com_off,
                rng=rng,
                level=level)

            for _ in range(n_envs):
                self.envs.append(MazeEnv(self, sampler, env_idx, bot_idx, text_mode))

                env_idx += 1
                bot_idx += sampler.n_bots

            self.level_slices.append((level, n_envs, slice(slc_idx, env_idx)))
            slc_idx = env_idx

            if level > self.max_level:
                self.max_level = level

        self.env_bot_idcs = np.array([env.bot_idx for env in self.envs])
        self.all_bot_idcs = np.concatenate([env.bot_idcs for env in self.envs])
        self.all_n_bots = np.array([env.sampler.n_bots for env in self.envs])
        self.n_all_bots = int(self.all_n_bots.sum())

        self.env_bot_handles = np.array([
            (env.handle, bot_handle)
            for env in self.envs
            for bot_handle in env.bot_handles], dtype=object)

    # --------------------------------------------------------------------------
    # MARK: textfill

    def textfill(self, n_aug: int, rng: 'None | int | np.random.Generator') -> 'dict[str, list[int]]':
        create_texture_from_file = self.gym.create_texture_from_file

        if self.text_mode == cfg.TEXT_NONE:
            textures = {}

        elif self.text_mode == cfg.TEXT_WARES:
            textures = {
                'whl': [create_texture_from_file(self.handle, cfg.ALT_ASSET_DIR + '/bot/wheel.png')],
                'cam': [create_texture_from_file(self.handle, cfg.ALT_ASSET_DIR + '/bot/camera.png')],
                'chs': [create_texture_from_file(self.handle, cfg.ALT_ASSET_DIR + '/bot/chassis.png')],
                'pld': [create_texture_from_file(self.handle, cfg.ALT_ASSET_DIR + '/bot/payload.png')],
                'bot': [create_texture_from_file(self.handle, cfg.ALT_ASSET_DIR + '/bot/body.png')],
                'obj': [create_texture_from_file(self.handle, cfg.ALT_ASSET_DIR + '/bot/crate.png')],
                'blk': [
                    create_texture_from_file(self.handle, os.path.join(cfg.ALT_ASSET_DIR, 'block', fname))
                    for fname in sorted(os.listdir(cfg.ALT_ASSET_DIR + '/block')) if fname[-1] == 'g'],
                'flr': [
                    create_texture_from_file(self.handle, os.path.join(cfg.ALT_ASSET_DIR, 'floor', fname))
                    for fname in sorted(os.listdir(cfg.ALT_ASSET_DIR + '/floor')) if fname[-1] == 'g'],
                'brd': [
                    create_texture_from_file(self.handle, os.path.join(cfg.ALT_ASSET_DIR, 'border', fname))
                    for fname in sorted(os.listdir(cfg.ALT_ASSET_DIR + '/border')) if fname[-1] == 'g']}

            textures['dec'] = textures['obj']
            textures['lnk'] = []

        elif self.text_mode == cfg.TEXT_ROOM:
            textures = {
                'whl': [
                    create_texture_from_file(self.handle, cfg.ALT_ASSET_DIR + '/bot/wheel.png'),
                    create_texture_from_file(self.handle, cfg.ASSET_DIR + '/other/wheel_alt.png')],
                'cam': [create_texture_from_file(self.handle, cfg.ALT_ASSET_DIR + '/bot/camera.png')],
                'bot': [
                    create_texture_from_file(self.handle, cfg.ASSET_DIR + '/other/paper01.png'),
                    create_texture_from_file(self.handle, cfg.ASSET_DIR + '/other/paper02.png')],
                'light': [
                    create_texture_from_file(self.handle, os.path.join(cfg.ASSET_DIR, 'light', fname))
                    for fname in sorted(os.listdir(cfg.ASSET_DIR + '/light'))],
                'dark': [
                    create_texture_from_file(self.handle, os.path.join(cfg.ASSET_DIR, 'dark', fname))
                    for fname in sorted(os.listdir(cfg.ASSET_DIR + '/dark'))],
                'brown': [
                    create_texture_from_file(self.handle, os.path.join(cfg.ASSET_DIR, 'brown', fname))
                    for fname in sorted(os.listdir(cfg.ASSET_DIR + '/brown'))],
                'pastel': [
                    create_texture_from_file(self.handle, os.path.join(cfg.ASSET_DIR, 'pastel', fname))
                    for fname in sorted(os.listdir(cfg.ASSET_DIR + '/pastel'))]}

            textures['chs'] = textures['bot'] + textures['light'] + textures['dark']
            textures['pld'] = textures['light'] + textures['dark']
            textures['obj'] = textures['light']
            textures['dec'] = textures['light'] + textures['dark']
            textures['blk'] = textures['brown'] + textures['light']
            textures['flr'] = textures['brown'] + textures['light'] + textures['dark'] + textures['pastel']
            textures['lnk'] = textures['light'] + textures['dark']
            textures['brd'] = []

            for k in ('light', 'dark', 'brown', 'pastel'):
                del textures[k]

        else:
            textures = {
                'whl': [
                    create_texture_from_file(self.handle, cfg.ALT_ASSET_DIR + '/bot/wheel.png'),
                    create_texture_from_file(self.handle, cfg.ASSET_DIR + '/other/wheel_alt.png')],
                'cam': [create_texture_from_file(self.handle, cfg.ALT_ASSET_DIR + '/bot/camera.png')],
                'bot': [
                    read_texture_file(cfg.ASSET_DIR + '/other/paper01.png'),
                    read_texture_file(cfg.ASSET_DIR + '/other/paper02.png')],
                'light': [
                    read_texture_file(os.path.join(cfg.ASSET_DIR, 'light', fname))
                    for fname in sorted(os.listdir(cfg.ASSET_DIR + '/light'))],
                'dark': [
                    read_texture_file(os.path.join(cfg.ASSET_DIR, 'dark', fname))
                    for fname in sorted(os.listdir(cfg.ASSET_DIR + '/dark'))],
                'brown': [
                    read_texture_file(os.path.join(cfg.ASSET_DIR, 'brown', fname))
                    for fname in sorted(os.listdir(cfg.ASSET_DIR + '/brown'))],
                'colour': [
                    read_texture_file(os.path.join(cfg.ASSET_DIR, 'colour', fname))
                    for fname in sorted(os.listdir(cfg.ASSET_DIR + '/colour'))],
                'pastel': [
                    read_texture_file(os.path.join(cfg.ASSET_DIR, 'pastel', fname))
                    for fname in sorted(os.listdir(cfg.ASSET_DIR + '/pastel'))]}

            textures['chs'] = textures['bot'] + textures['light'] + textures['dark']
            textures['pld'] = textures['light'] + textures['dark']
            textures['bot'] = textures['bot'] + textures['light'] + textures['dark']
            textures['obj'] = textures['light']
            textures['dec'] = textures['light'] + textures['dark']
            textures['blk'] = textures['brown'] + textures['light'] + textures['colour'] + textures['pastel']
            textures['flr'] = textures['brown'] + textures['light'] + textures['dark'] + textures['pastel']
            textures['lnk'] = textures['light'] + textures['dark']
            textures['brd'] = []

            for k in ('light', 'dark', 'brown', 'colour', 'pastel'):
                del textures[k]

            decal = read_texture_file(cfg.ASSET_DIR + '/other/paper03.jpg', False)
            rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

            if self.text_mode == cfg.TEXT_RANDAUG:
                for k in set(textures) - set(('whl', 'cam', 'brd')):
                    textures[k] = self.augment(textures[k], decal, n_aug, rng, cfg.AUG_DECAL)

            else:
                n_aug_decal = max(n_aug // 2, 1)
                n_aug_base = max(n_aug - n_aug_decal - 1, 1)

                for k in set(textures) - set(('whl', 'cam', 'brd')):
                    textures[k] = (
                        self.augment(textures[k], decal, 1, rng, cfg.AUG_NONE) +
                        self.augment(textures[k], decal, n_aug_base, rng, cfg.AUG_BASE) +
                        self.augment(textures[k], decal, n_aug_decal, rng, cfg.AUG_DECAL))

        return textures

    # --------------------------------------------------------------------------
    # MARK: augment

    def augment(
        self,
        imgs: 'list[ndarray]',
        decal: ndarray,
        n_aug: int,
        rng: np.random.Generator,
        aug_mode: int,
        brightness: 'tuple[float, float]' = (-0.2, 0.2),
        contrast: 'tuple[float, float]' = (0.8, 1.2),
        decal_opacity: 'tuple[float, float]' = (0.7, 1.)
    ) -> 'list[ndarray]':
        tlist = []

        for _ in range(n_aug):
            for img in imgs:
                if aug_mode != cfg.AUG_NONE:
                    img_ = warp_basic(img, rng)

                    if aug_mode == cfg.AUG_DECAL:
                        overlay = rng.uniform(*decal_opacity) * warp_shear(decal, rng, img_.shape)[..., None]
                        img_ = (1. - overlay) * img_ + overlay * rng.choice(cfg.COLOURS['wall'])

                    img_ = np.clip(img_ * rng.uniform(*contrast) + rng.uniform(*brightness), 0., 1.)

                else:
                    img_ = img

                h, w, c = img_.shape
                w4 = w * (c + 1)

                alpha = np.full((h, w, 1), 255, dtype=np.uint8)
                img_ = (img_ * 255.).astype(np.uint8)
                img_ = np.concatenate((img_, alpha), axis=-1)

                tlist.append(self.gym.create_texture_from_buffer(self.handle, w, h, img_.reshape((h, w4))))

        return tlist

    # --------------------------------------------------------------------------
    # MARK: relight

    def relight(self, rng: np.random.Generator = None):
        """Override lighting (intensity, ambient, direction)."""

        n_lights = len(self.LIGHT_DIRECTIONS)

        if rng is not None and rng.random() < cfg.LIGHT_RAND_PROB:
            lint_min = cfg.BASE_LIGHT_INTENSITY - cfg.MAX_INTENSITY_OFFSET
            lints = lint_min + 2 * cfg.MAX_INTENSITY_OFFSET * rng.random(n_lights)

            lamb_min = cfg.BASE_LIGHT_AMBIENT - cfg.MAX_AMBIENT_OFFSET
            lambs = lamb_min + 2 * cfg.MAX_AMBIENT_OFFSET * rng.random(n_lights)

            for i, ldir, lint, lamb in zip(range(n_lights), self.LIGHT_DIRECTIONS, lints, lambs):
                self.gym.set_light_parameters(
                    self.handle, i, gymapi.Vec3(lint, lint, lint), gymapi.Vec3(lamb, lamb, lamb), ldir)

        else:
            lint = cfg.BASE_LIGHT_INTENSITY
            lamb = cfg.BASE_LIGHT_AMBIENT

            for i, ldir in enumerate(self.LIGHT_DIRECTIONS):
                self.gym.set_light_parameters(
                    self.handle, i, gymapi.Vec3(lint, lint, lint), gymapi.Vec3(lamb, lamb, lamb), ldir)

    # --------------------------------------------------------------------------
    # MARK: reset

    def reset(self, env_rst_idcs: 'list[int]', full: bool) -> 'tuple[ndarray, ndarray]':
        """Reset envs. and collect the new states."""

        # Randomise global lighting when the highest-level env. is reset
        for i in env_rst_idcs:
            sampler = self.envs[i].sampler

            if sampler.level == self.max_level:
                self.relight(sampler.rng)
                break

        actor_states = [self.envs[i].reset(full) for i in env_rst_idcs]

        env_idx_keys = ('wall_idcs', 'blk_idcs', 'link_idcs', 'obj_idcs', 'decoy_idcs', 'bot_idcs')
        actor_idcs = [getattr(self.envs[i], k) for i in env_rst_idcs for k in env_idx_keys]

        return np.concatenate(actor_states), np.concatenate(actor_idcs)

    # --------------------------------------------------------------------------
    # MARK: cleanup

    def cleanup(self):
        """Destroy existing envs. and the sim., along with their data."""

        for env in self.envs:
            for cam_handle in env.cam_handles:
                self.gym.destroy_camera_sensor(self.handle, env.handle, cam_handle)

            self.gym.destroy_env(env.handle)

        self.envs.clear()
        self.gym.destroy_sim(self.handle)
