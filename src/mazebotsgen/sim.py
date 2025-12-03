"""Gym wrappers"""

import os
from argparse import Namespace

import numpy as np
from numpy import ndarray
from scipy.spatial.transform import Rotation
from isaacgym import gymapi

import config as cfg
from maze import MazeConstructor
from utils import get_cached_paths, get_numba_dict


# ------------------------------------------------------------------------------
# MARK: MazeEnv

class MazeEnv:
    """`Gym.Env` wrapper for env. setup and path estimation."""

    NEUTRAL_CLR = gymapi.Vec3(1., 1., 1.)
    GOAL_CLRS = np.array([gymapi.Vec3(*clr) for clr in cfg.COLOURS['goal']], dtype=object)

    MAX_SIDE_LENGTH = cfg.LEVEL_PARAMS[max(cfg.LEVEL_PARAMS)]['side_length']

    FLOOR_POSE = (0., 0., -cfg.WALL_WIDTH / 2.)
    BORDER_POSES = {}

    for lvl, params in cfg.LEVEL_PARAMS.items():
        border_length = params['side_length'] + cfg.WALL_WIDTH
        border_halflength = border_length / 2.

        BORDER_POSES[lvl] = [
            ((-border_halflength, 0., cfg.BLOCK_HALFHEIGHT), 0.),
            ((0., -border_halflength, cfg.BLOCK_HALFHEIGHT), np.pi/2.),
            ((border_halflength, 0., cfg.BLOCK_HALFHEIGHT), 0.),
            ((0., border_halflength, cfg.BLOCK_HALFHEIGHT), np.pi/2.)]

    blk_handles: ndarray
    obj_handles: ndarray
    bot_handles: ndarray
    cam_handles: ndarray

    blk_idcs: ndarray
    obj_idcs: ndarray
    bot_idcs: ndarray

    data: 'dict[str, ndarray]'
    cell_pass_map: 'dict[int, ndarray]'

    def __init__(self, sim: 'MazeSim', sampler: MazeConstructor, idx: int, bot_idx: int):
        self.sim = sim
        self.sampler = sampler
        self.idx = idx
        self.bot_idx = bot_idx
        self.bot_slice = slice(bot_idx, bot_idx + sampler.n_bots)

        # Place environment onto the grid
        bbox_side_halflength = self.MAX_SIDE_LENGTH / 2. + cfg.ENV_HALFSPACING

        bbox_vertex_low = gymapi.Vec3(-bbox_side_halflength, -bbox_side_halflength, 0.)
        bbox_vertex_high = gymapi.Vec3(bbox_side_halflength, bbox_side_halflength, cfg.WALL_HEIGHT)

        self.handle = sim.gym.create_env(
            sim.handle,
            bbox_vertex_low,
            bbox_vertex_high,
            sim.n_envs_per_row)

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
        n_cells = np.prod(sampler.cell_pt_grid.shape[:-1])

        self.blk_substate = np.concatenate((
            np.zeros((n_cells, 3), dtype=np.float32),
            np.ones((n_cells, 1), dtype=np.float32),
            np.zeros((n_cells, 6), dtype=np.float32)), axis=-1, dtype=np.float32)

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
        lvl = self.sampler.level

        self.base_handles = np.zeros(5, dtype=np.int32)
        self.blk_handles = np.zeros(self.sampler.cell_pt_grid.shape[:2], dtype=np.int32)
        self.obj_handles = np.zeros(self.sampler.n_goals, dtype=np.int32)
        self.decoy_handles = np.zeros(self.sampler.n_decoys, dtype=np.int32)

        self.base_idcs = np.zeros(5, dtype=np.int32)
        self.blk_idcs = np.zeros(self.sampler.cell_pt_grid.shape[:2], dtype=np.int32)
        self.obj_idcs = np.zeros(self.sampler.n_goals, dtype=np.int32)
        self.decoy_idcs = np.zeros(self.sampler.n_decoys, dtype=np.int32)

        self.base_handles[0] = floor_handle = gym.create_actor(
            self.handle,
            self.sim.assets['floor'],
            gymapi.Transform(gymapi.Vec3(*self.FLOOR_POSE)),
            'floor',
            self.idx,
            -1,
            cfg.ENT_CLS_FLOOR)

        self.base_idcs[0] = gym.get_actor_index(self.handle, floor_handle, gymapi.DOMAIN_SIM)

        for i in range(4):
            pose, angle = self.BORDER_POSES[lvl][i]

            self.base_handles[i+1] = border_handle = gym.create_actor(
                self.handle,
                self.sim.assets['border'],
                gymapi.Transform(gymapi.Vec3(*pose), gymapi.Quat.from_euler_zyx(0., 0., angle)),
                f'border-{i:02d}',
                self.idx,
                -1,
                cfg.ENT_CLS_WALL)

            self.base_idcs[i+1] = gym.get_actor_index(self.handle, border_handle, gymapi.DOMAIN_SIM)

        for i in range(self.sampler.n_side_divs):
            for j in range(self.sampler.n_side_divs):
                pose = gymapi.Transform(gymapi.Vec3(
                    *self.sampler.cell_pt_grid[i, j],
                    cfg.BLOCK_HALFHEIGHT if self.data.block_mask[i, j] else cfg.BLOCK_HIDDENHEIGHT))

                self.blk_handles[i, j] = blk_handle = gym.create_actor(
                    self.handle,
                    self.sim.assets['block'],
                    pose,
                    f'block-{i:02d}-{j:02d}',
                    self.idx,
                    -1,
                    cfg.ENT_CLS_WALL)

                self.blk_idcs[i, j] = gym.get_actor_index(self.handle, blk_handle, gymapi.DOMAIN_SIM)

        self.blk_idcs = self.blk_idcs.flatten()

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
        # camera_props.horizontal_fov = 90.         # Default
        camera_props.far_plane = 64.                # 2e+6 by default
        camera_pos = gymapi.Vec3(*cfg.CAM_OFFSET)

        self.cam_handles = np.zeros_like(self.bot_handles)

        for i, bot_handle in enumerate(self.bot_handles):
            self.cam_handles[i] = cam_handle = gym.create_camera_sensor(self.handle, camera_props)
            body_handle = gym.get_actor_rigid_body_handle(self.handle, bot_handle, cfg.BOT_BODY_IDX)

            gym.attach_camera_to_body(
                cam_handle, self.handle, body_handle, gymapi.Transform(camera_pos), gymapi.FOLLOW_TRANSFORM)

    # --------------------------------------------------------------------------
    # MARK: recolour

    def recolour(self):
        """Set obj. and bot colours according to associated goals."""

        rng = self.sampler.rng
        set_txt = self.sim.gym.set_rigid_body_texture
        set_clr = self.sim.gym.set_rigid_body_color
        set_sid = self.sim.gym.set_rigid_body_segmentation_id

        # Objects
        obj_clrs = self.GOAL_CLRS[self.data.obj_goal_map]

        for obj_handle, clr, clr_idx in zip(self.obj_handles, obj_clrs, self.data.obj_goal_map):
            set_txt(self.handle, obj_handle, 0, gymapi.MESH_VISUAL, self.sim.textures['obj'])
            set_clr(self.handle, obj_handle, 0, gymapi.MESH_VISUAL, clr)
            set_sid(self.handle, obj_handle, 0, cfg.OBJ_CLS_OFFSET + clr_idx)

        # Clutter
        for decoy_handle in self.decoy_handles:
            set_txt(self.handle, decoy_handle, 0, gymapi.MESH_VISUAL, self.sim.textures['obj'])

        # Bots
        body_clrs = self.GOAL_CLRS[self.data.bot_goal_map]

        for bot_handle, clr, clr_idx in zip(self.bot_handles, body_clrs, self.data.bot_goal_map):
            set_txt(self.handle, bot_handle, cfg.BOT_BODY_IDX, gymapi.MESH_VISUAL, self.sim.textures['bot'])
            set_txt(self.handle, bot_handle, cfg.BOT_LOAD_IDX, gymapi.MESH_VISUAL, self.sim.textures['pld'])
            set_txt(self.handle, bot_handle, 2, gymapi.MESH_VISUAL, self.sim.textures['cam'])
            set_txt(self.handle, bot_handle, 3, gymapi.MESH_VISUAL, self.sim.textures['chs'])

            for i in range(4):
                set_txt(self.handle, bot_handle, 4+i, gymapi.MESH_VISUAL, self.sim.textures['whl'])

            set_clr(self.handle, bot_handle, cfg.BOT_LOAD_IDX, gymapi.MESH_VISUAL, clr)
            set_sid(self.handle, bot_handle, cfg.BOT_LOAD_IDX, cfg.BOT_CLS_OFFSET + clr_idx)

        # Base
        set_txt(self.handle, self.base_handles[0], 0, gymapi.MESH_VISUAL, rng.choice(self.sim.textures['flr']))
        brd_txt = rng.choice(self.sim.textures['brd'])

        for border_handle in self.base_handles[1:]:
            set_txt(self.handle, border_handle, 0, gymapi.MESH_VISUAL, brd_txt)

        # Blocks
        for i in range(self.blk_handles.shape[0]):
            for j in range(self.blk_handles.shape[1]):
                handle = self.blk_handles[i, j]
                clr_idx = max(self.data.cell_clr_idx_grid[i, j], 0)

                set_txt(self.handle, handle, 0, gymapi.MESH_VISUAL, self.sim.textures['blk'][clr_idx])
                set_sid(self.handle, handle, 0, cfg.WALL_CLS_OFFSET + clr_idx)

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

        blk_states = np.concatenate((
            self.cell_pts,
            np.where(
                self.data.block_mask.ravel(),
                cfg.BLOCK_HALFHEIGHT,
                cfg.BLOCK_HIDDENHEIGHT).astype(np.float32)[:, None],
            self.blk_substate), axis=-1, dtype=np.float32)

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

        actor_states = np.concatenate((blk_states, obj_states, decoy_states, bot_states))

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
        bot_asset_options.use_mesh_materials = True

        static_asset_options = gymapi.AssetOptions()
        static_asset_options.thickness = 0.005
        static_asset_options.use_mesh_materials = True
        static_asset_options.fix_base_link = True
        static_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX

        self.assets = {
            'bot': gym.load_urdf(self.handle, cfg.ASSET_DIR, 'mazebot_v2.urdf', bot_asset_options),
            'obj': gym.load_urdf(self.handle, cfg.ASSET_DIR, 'cube.urdf', static_asset_options),
            'block': gym.load_asset(self.handle, cfg.ASSET_DIR, 'block.urdf', static_asset_options),
            'border': gym.load_urdf(self.handle, cfg.ASSET_DIR, 'border.urdf', static_asset_options),
            'floor': gym.load_urdf(self.handle, cfg.ASSET_DIR, 'floor.urdf', static_asset_options)}

        self.textures = {
            'whl': gym.create_texture_from_file(self.handle, cfg.ASSET_DIR + '/bot/wheel.png'),
            'cam': gym.create_texture_from_file(self.handle, cfg.ASSET_DIR + '/bot/camera.png'),
            'chs': gym.create_texture_from_file(self.handle, cfg.ASSET_DIR + '/bot/chassis.png'),
            'pld': gym.create_texture_from_file(self.handle, cfg.ASSET_DIR + '/bot/payload.png'),
            'bot': gym.create_texture_from_file(self.handle, cfg.ASSET_DIR + '/bot/body.png'),
            'obj': gym.create_texture_from_file(self.handle, cfg.ASSET_DIR + '/bot/crate.png'),
            'blk': [
                gym.create_texture_from_file(self.handle, os.path.join(cfg.ASSET_DIR, 'block', fname))
                for fname in sorted(os.listdir(cfg.ASSET_DIR + '/block')) if fname[-1] == 'g'],
            'flr': [
                gym.create_texture_from_file(self.handle, os.path.join(cfg.ASSET_DIR, 'floor', fname))
                for fname in sorted(os.listdir(cfg.ASSET_DIR + '/floor')) if fname[-1] == 'g'],
            'brd': [
                gym.create_texture_from_file(self.handle, os.path.join(cfg.ASSET_DIR, 'border', fname))
                for fname in sorted(os.listdir(cfg.ASSET_DIR + '/border')) if fname[-1] == 'g']}

        # Override for const. num. training
        if '+' in args.env_cfg:
            params = cfg.LEVEL_PARAMS[int(args.env_cfg.split('+')[0].split('x')[1])]
            n_bots_override = params['n_bots']
            n_goals_override = params['n_goals']

            for level in tuple(cfg.LEVEL_PARAMS):
                params = cfg.LEVEL_PARAMS[level]
                params['n_bots'] = n_bots_override
                params['n_goals'] = n_goals_override

        # Create parallel envs.
        self.n_envs = sum(n_envs for n_envs in cfg.ENV_NUM_LVL_PRESETS[args.env_cfg].values())
        self.n_envs_per_row: int = round(self.n_envs**0.5)

        self.max_level = min(cfg.LEVEL_PARAMS)
        self.level_slices = []
        self.envs = []
        env_idx = 0
        bot_idx = 0
        slc_idx = 0

        for level, n_envs in cfg.ENV_NUM_LVL_PRESETS[args.env_cfg].items():
            sampler = MazeConstructor(
                **cfg.LEVEL_PARAMS[level],
                n_decoys=cfg.LEVEL_PARAMS[level]['n_goals'] * (level if args.clutter_fn == 'prog' else 1),
                global_spawn_prob=args.global_spawn_prob,
                speaker_dropout=args.prob_com_off,
                rng=rng,
                level=level)

            for _ in range(n_envs):
                self.envs.append(MazeEnv(self, sampler, env_idx, bot_idx))

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

        env_idx_keys = ('blk_idcs', 'obj_idcs', 'decoy_idcs', 'bot_idcs')
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
