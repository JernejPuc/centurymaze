"""Gym wrappers"""

from argparse import Namespace

import numpy as np
from numpy import ndarray
from scipy.spatial.transform import Rotation
from isaacgym import gymapi

import config as cfg
from maze import MazeSampler
from utils import get_cached_paths, get_numba_dict


# ------------------------------------------------------------------------------
# MARK: MazeEnv

class MazeEnv:
    """`Gym.Env` wrapper for env. setup and path estimation."""

    GOAL_CLRS = np.array([gymapi.Vec3(*clr) for clr in cfg.COLOURS['goal']], dtype=object)
    BEACON_OFF_CLR = gymapi.Vec3(*cfg.COLOURS['beacon'][0])

    box_handles: ndarray
    obj_handles: ndarray
    bot_handles: ndarray
    cam_handles: ndarray

    box_idcs: ndarray
    obj_idcs: ndarray
    bot_idcs: ndarray

    obj_pts: ndarray
    spawn_pts: ndarray
    spawn_angles: ndarray
    bot_obj_map: ndarray
    obj_goal_map: ndarray
    bot_goal_map: ndarray

    cellpair_path_map: 'dict[tuple[int, int], ndarray]'

    def __init__(self, sim: 'MazeSim', idx: int = 0):
        self.sim = sim
        self.idx = idx

        # Place environment onto the grid
        bbox_side_halflength = sim.data['spec'].item()['grid']['side_length'] / 2 + cfg.ENV_HALFSPACING

        bbox_vertex_low = gymapi.Vec3(-bbox_side_halflength, -bbox_side_halflength, 0.)
        bbox_vertex_high = gymapi.Vec3(bbox_side_halflength, bbox_side_halflength, 1.)

        self.handle = sim.gym.create_env(
            sim.handle,
            bbox_vertex_low,
            bbox_vertex_high,
            sim.n_envs_per_row)

        # Place entities into the environment
        self.resample()
        self.create_layout()
        self.create_objects()
        self.create_bots()
        self.recolour()

        # Prep. for path estimation
        self.cell_pts = sim.data['cell_pt_grid'].reshape(-1, 2)
        self.cell_wall_grid = sim.data['cell_wall_grid']
        self.open_delims = sim.sampler.open_delims

        # Convert passage map into numba typed dict.
        self.cell_pass_map = get_numba_dict()

        for key, lst in sim.data['cell_pass_map'].item().items():
            self.cell_pass_map[key] = np.array(lst, dtype=np.int64)

    # --------------------------------------------------------------------------
    # MARK: resample

    def resample(self):
        """Reinit. the path map, points, and goal distribution."""

        self.cellpair_path_map = get_numba_dict(tuple_as_key=True)

        self.obj_pts, self.spawn_pts, self.spawn_angles, self.bot_obj_map, self.obj_goal_map, self.bot_goal_map = \
            self.sim.sampler.sample_tasks()

    # --------------------------------------------------------------------------
    # MARK: create_layout

    def create_layout(self):
        """Place static boxes according to the layout spec."""

        wall_clrs = [gymapi.Vec3(*clr) for clr in cfg.COLOURS['wall']]
        floor_clr = gymapi.Vec3(*cfg.COLOURS['background'][cfg.FLOOR_CLR_IDX])

        box_handles = []
        box_idcs = []

        for name, (pos_x, pos_y, pos_z, asset_key, clr_idx) in self.sim.data['spec'].item()['layout'].items():
            pose = gymapi.Transform(gymapi.Vec3(pos_x, pos_y, pos_z))

            # Switch between floor and wall colour & tag
            if clr_idx < 0:
                clr = floor_clr
                ent_cls = cfg.ENT_CLS_FLOOR

            else:
                clr = wall_clrs[clr_idx]
                ent_cls = cfg.ENT_CLS_WALL

            box_handles.append(box_handle := self.sim.gym.create_actor(
                    self.handle,
                    self.sim.assets[asset_key],
                    pose,
                    name,
                    self.idx,
                    -1,
                    ent_cls))

            box_idcs.append(self.sim.gym.get_actor_index(self.handle, box_handle, gymapi.DOMAIN_SIM))

            self.sim.gym.set_rigid_body_color(self.handle, box_handle, cfg.OBJ_BODY_IDX, gymapi.MESH_VISUAL, clr)

        self.box_handles = np.array(box_handles)
        self.box_idcs = np.array(box_idcs)

    # --------------------------------------------------------------------------
    # MARK: create_objects

    def create_objects(self):
        """Place goal objects at target locations."""

        self.obj_handles = np.zeros(self.sim.n_goals, dtype=np.int32)
        self.obj_idcs = np.zeros(self.sim.n_goals, dtype=np.int32)

        for i, pos in enumerate(self.obj_pts):
            pose = gymapi.Transform(gymapi.Vec3(*pos, cfg.OBJ_HEIGHT))

            self.obj_handles[i] = obj_handle = self.sim.gym.create_actor(
                self.handle,
                self.sim.assets['obj'],
                pose,
                f'object-{i:d}',
                self.idx,
                -1,
                cfg.ENT_CLS_OBJ)

            self.obj_idcs[i] = self.sim.gym.get_actor_index(self.handle, obj_handle, gymapi.DOMAIN_SIM)

    # --------------------------------------------------------------------------
    # MARK: create_bots

    def create_bots(self):
        """Place agents at spawn points with set DOF and camera properties."""

        gym = self.sim.gym

        self.bot_handles = np.zeros(self.sim.n_bots, dtype=np.int32)
        self.bot_idcs = np.zeros(self.sim.n_bots, dtype=np.int32)

        for i, pos, angle in zip(range(self.sim.n_bots), self.spawn_pts, self.spawn_angles):
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
                    cfg.ENT_CLS_CHASSIS)

            self.bot_idcs[i] = gym.get_actor_index(self.handle, bot_handle, gymapi.DOMAIN_SIM)

            gym.set_rigid_body_segmentation_id(self.handle, bot_handle, cfg.BOT_BODY_IDX, cfg.ENT_CLS_BODY)
            gym.set_rigid_body_segmentation_id(self.handle, bot_handle, cfg.BOT_BEACON_IDX, cfg.ENT_CLS_BEACON)

        # Set bot motor params. and rigid shape interaction props.
        dof_props: 'dict[str, ndarray]' = gym.get_actor_dof_properties(self.handle, bot_handle)
        dof_props['driveMode'].fill(gymapi.DOF_MODE_EFFORT)
        dof_props['stiffness'].fill(cfg.MOT_STIFFNESS)
        dof_props['damping'].fill(cfg.MOT_DAMPING)

        for bot_handle in self.bot_handles:
            gym.set_actor_dof_properties(self.handle, bot_handle, dof_props)

            shape_prop_list = gym.get_actor_rigid_shape_properties(self.handle, bot_handle)

            for shape_props in shape_prop_list:
                shape_props.friction = 0.
                # shape_props.rolling_friction
                # shape_props.torsion_friction

            gym.set_actor_rigid_shape_properties(self.handle, bot_handle, shape_prop_list)

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

        set_clr = self.sim.gym.set_rigid_body_color

        # Objects
        obj_clrs = self.GOAL_CLRS[self.obj_goal_map]

        for obj_handle, clr in zip(self.obj_handles, obj_clrs):
            set_clr(self.handle, obj_handle, cfg.OBJ_BODY_IDX, gymapi.MESH_VISUAL, clr)

        # Bots
        body_clrs = self.GOAL_CLRS[self.bot_goal_map]

        for bot_handle, clr in zip(self.bot_handles, body_clrs):
            set_clr(self.handle, bot_handle, cfg.BOT_BODY_IDX, gymapi.MESH_VISUAL, clr)
            set_clr(self.handle, bot_handle, cfg.BOT_BEACON_IDX, gymapi.MESH_VISUAL, self.BEACON_OFF_CLR)

    # --------------------------------------------------------------------------
    # MARK: reset

    def reset(self) -> ndarray:
        """Set actors to new starting conditions and relay the new states."""

        self.resample()
        self.recolour()

        obj_states = np.concatenate((
            self.obj_pts,
            self.sim.obj_substate), axis=-1, dtype=np.float32)

        bot_states = np.concatenate((
            self.spawn_pts,
            self.sim.bot_substates[0],
            Rotation.from_euler('z', self.spawn_angles).as_quat(),
            self.sim.bot_substates[1]), axis=-1, dtype=np.float32)

        actor_states = np.concatenate((obj_states, bot_states))

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
        n_envs=1,
        n_bots=-1,
        n_goals=-1,
        global_spawn_prob=0.,
        draw_freq=64,
        use_gpu_pipeline=True,
        use_gpu=True,
        num_threads=0,
        compute_device_id=0,
        graphics_device_id=0)

    def __init__(self, args: Namespace = DEFAULT_SIM_ARGS, rng: 'None | int | np.random.Generator' = None):

        # Init. Isaac Gym
        self.gym = gym = gymapi.acquire_gym()

        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0., 0., -9.80665)
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
        if args.use_gpu_pipeline:
            if args.graphics_device_id != args.compute_device_id:
                print(f'Warning: Overriding graphics device {args.graphics_device_id} to {args.compute_device_id}.')

                args.graphics_device_id = args.compute_device_id

        # Init. PhysX
        self.handle = gym.create_sim(args.compute_device_id, args.graphics_device_id, gymapi.SIM_PHYSX, sim_params)

        if self.handle is None:
            raise Exception('Failed to initialise.')

        # Override lighting to normalise viewing conditions
        # 0, (1., 1., 1.), (0.1, 0.1, 0.1), (1., 1., 4.)
        # 1, (0.5, 0.5, 0.5), (0.1, 0.1, 0.1), (1., -1., 4.)
        # 2, (0., 0., 0.), (0., 0., 0.), (0., 0., 0.)
        # 3, (0., 0., 0.), (0., 0., 0.), (0., 0., 0.)
        gym.set_light_parameters(
            self.handle, 0, gymapi.Vec3(0.2, 0.2, 0.2), gymapi.Vec3(0.8, 0.8, 0.8), gymapi.Vec3(1., 1., 2.))
        gym.set_light_parameters(
            self.handle, 1, gymapi.Vec3(0.2, 0.2, 0.2), gymapi.Vec3(0.8, 0.8, 0.8), gymapi.Vec3(1., -1., 2.))
        gym.set_light_parameters(
            self.handle, 2, gymapi.Vec3(0.2, 0.2, 0.2), gymapi.Vec3(0.8, 0.8, 0.8), gymapi.Vec3(-1., 0.33, 2.))
        gym.set_light_parameters(
            self.handle, 3, gymapi.Vec3(0.2, 0.2, 0.2), gymapi.Vec3(0.8, 0.8, 0.8), gymapi.Vec3(-1., -0.33, 2.))

        # Load structural data/spec.
        self.data = np.load(cfg.ASSET_DIR + '/maze_data.npz', allow_pickle=True)

        # Load and create assets
        placed_asset_options = gymapi.AssetOptions()
        placed_asset_options.thickness = 0.005  # 0.02 default; change not visible, but this relates better to bot size

        static_asset_options = gymapi.AssetOptions()
        static_asset_options.fix_base_link = True

        self.assets = {
            'bot': gym.load_urdf(self.handle, cfg.ASSET_DIR, cfg.BOT_FILE_NAME, placed_asset_options),
            'obj': gym.load_urdf(self.handle, cfg.ASSET_DIR, cfg.OBJ_FILE_NAME, placed_asset_options)}

        for key, dims in self.data['spec'].item()['assets'].items():
            self.assets[key] = gym.create_box(self.handle, *dims, static_asset_options)

        # Set env. & sampling params.
        self.n_goals = args.n_goals if args.n_goals > 0 else cfg.N_GOAL_CLRS
        self.n_bots = args.n_bots if args.n_bots > 0 else len(self.data['spawn_pts'])
        self.n_envs = args.n_envs
        self.n_envs_per_row: int = round(self.n_envs**0.5)
        self.n_all_bots: int = self.n_envs * self.n_bots
        self.n_bots_per_goal = self.n_bots / self.n_goals

        self.sampler = MazeSampler(self.data, self.n_bots, self.n_goals, args.global_spawn_prob, rng)

        # Create parallel envs.
        self.envs = [MazeEnv(self, idx) for idx in range(self.n_envs)]

        self.env_bot_handles = np.array([
            (env.handle, bot_handle)
            for env in self.envs
            for bot_handle in env.bot_handles], dtype=object)

        # Data for env. resetting
        self.obj_substate = np.concatenate((
            np.ones((self.n_goals, 1), dtype=np.float32) * cfg.OBJ_HEIGHT,
            np.zeros((self.n_goals, 3), dtype=np.float32),
            np.ones((self.n_goals, 1), dtype=np.float32),
            np.zeros((self.n_goals, 6), dtype=np.float32)), axis=-1, dtype=np.float32)

        self.bot_substates = (
            np.zeros((self.n_bots, 1), dtype=np.float32),
            np.zeros((self.n_bots, 6), dtype=np.float32))

    # --------------------------------------------------------------------------
    # MARK: reset

    def reset(self, env_rst_idcs: 'list[int]') -> 'tuple[ndarray, ndarray]':
        """Reset envs. and collect the new states."""

        actor_states = [self.envs[i].reset() for i in env_rst_idcs]

        env_idx_keys = ('obj_idcs', 'bot_idcs')
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
