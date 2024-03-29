import json
from argparse import Namespace

import numpy as np
from numpy import ndarray
from scipy.spatial.transform import Rotation
from isaacgym import gymapi

import config as cfg
from maze import MazeData, MazeConstructor


# Assets
BOT_FILE_NAME = 'mazebot.urdf'
OBJ_FILE_NAME = 'levisphere.urdf'

OBJECT_BODY_IDX = 0
BOT_BODY_IDX = 0
BOT_CARGO_IDX = 1

MOT_STIFFNESS = 0.001   # Bot won't move at 0.1, slow turning difficult at 0.01
MOT_DAMPING = 0.02      # Applying torque without damping makes the bot fly off
MOT_MAX_TORQUE = 1.     # Per motor/wheel

# Lowered inide the body, because the cargo was clipping in any configuration
CAM_OFFSET = np.array([cfg.BOT_WIDTH/2. - 0.025, 0., cfg.BOT_HEIGHT - 0.1])


class MazeEnv:
    """
    Gym.Env wrapper for mazes inhabited by 4-wheeled robotic agents and static entities,
    i.e. walls and uniquely coloured objects.
    """

    ENV_HALFSPACING = 0.1

    WALL_COLOURS = np.array([gymapi.Vec3(*clr) for clr in cfg.COLOURS['pastel']], dtype=object)
    OBJECT_COLOURS = np.array([gymapi.Vec3(*clr) for clr in cfg.COLOURS['basic']], dtype=object)
    LINK_COLOUR = gymapi.Vec3(*cfg.COLOURS['grey'][2])
    BASE_BOT_COLOUR = gymapi.Vec3(*cfg.COLOURS['grey'][1])
    BASE_CARGO_COLOUR = gymapi.Vec3(*cfg.COLOURS['grey'][1])

    box_handles: ndarray
    wally_handles: ndarray
    wallx_handles: ndarray
    link_handles: ndarray
    roof_handles: ndarray
    obj_handles: ndarray
    bot_handles: ndarray
    cam_handles: ndarray

    box_indices: ndarray
    wally_indices: ndarray
    wallx_indices: ndarray
    link_indices: ndarray
    roof_indices: ndarray
    obj_indices: ndarray
    bot_indices: ndarray

    def __init__(self, sim: 'MazeSim', data: MazeData, group_id: int = 0):
        self.sim = sim
        self.data = data
        self.group_id = group_id

        # Place environment onto the grid
        env_bbox_halfwidth = sim.constructor.supenv_halfwidth + self.ENV_HALFSPACING

        env_bbox_vertex_low = gymapi.Vec3(-env_bbox_halfwidth, -env_bbox_halfwidth, 0.)
        env_bbox_vertex_high = gymapi.Vec3(env_bbox_halfwidth, env_bbox_halfwidth, 1.)

        self.handle = sim.gym.create_env(
            sim.handle,
            env_bbox_vertex_low,
            env_bbox_vertex_high,
            sim.n_envs_per_row)

        # Place entities (actors) into the environment
        self.init_static() if not sim.is_preset else self.init_preset()
        self.init_agents()
        self.set_colours()
        self.set_rigid_props()

    def init_preset(self):
        """Place objects and preset boxes."""

        sim = self.sim
        gym = sim.gym
        data = self.data

        self.box_handles = []
        self.box_indices = []
        self.obj_handles = np.zeros(len(data.obj_clr_idcs), dtype=np.int32)
        self.obj_indices = np.zeros(len(data.obj_clr_idcs), dtype=np.int32)

        # Presets
        for name, (pos_x, pos_y, asset_key, clr_idx) in sim.preset_specs.items():
            asset, pos_z = sim.preset_assets[asset_key]

            if clr_idx < 0:
                pos_z = -pos_z
                clr = self.LINK_COLOUR
                seg_class = cfg.SEG_CLS_PLANE

            else:
                clr = self.WALL_COLOURS[clr_idx]
                seg_class = cfg.SEG_CLS_WALL

            self.box_handles.append(handle := gym.create_actor(
                    self.handle,
                    asset,
                    gymapi.Transform(gymapi.Vec3(pos_x, pos_y, pos_z)),
                    name,
                    self.group_id,
                    -1,
                    seg_class))

            self.box_indices.append(gym.get_actor_index(self.handle, handle, gymapi.DOMAIN_SIM))

            gym.set_rigid_body_color(self.handle, handle, OBJECT_BODY_IDX, gymapi.MESH_VISUAL, clr)

        # Objectives
        for i, pos in enumerate(data.obj_points):
            pose = gymapi.Transform(gymapi.Vec3(*pos, cfg.OBJECT_HEIGHT))

            self.obj_handles[i] = handle = gym.create_actor(
                self.handle,
                sim.asset_object,
                pose,
                f'object-{i:02d}',
                self.group_id,
                -1,
                cfg.SEG_CLS_OBJ)

            self.obj_indices[i] = gym.get_actor_index(self.handle, handle, gymapi.DOMAIN_SIM)

    def init_static(self):
        """Place objects and raise or lower walls and links wrt. associated masks."""

        sim = self.sim
        gym = sim.gym
        cons = sim.constructor
        data = self.data

        self.wally_handles = np.zeros(cons.hor_grid.shape[:2], dtype=np.int32)
        self.wallx_handles = np.zeros(cons.ver_grid.shape[:2], dtype=np.int32)
        self.link_handles = np.zeros(cons.link_grid.shape[:2], dtype=np.int32)
        self.roof_handles = np.zeros(cons.roof_grid.shape[:2], dtype=np.int32)
        self.obj_handles = np.zeros(len(data.obj_clr_idcs), dtype=np.int32)

        self.wally_indices = np.zeros(cons.hor_grid.shape[:2], dtype=np.int32)
        self.wallx_indices = np.zeros(cons.ver_grid.shape[:2], dtype=np.int32)
        self.link_indices = np.zeros(cons.link_grid.shape[:2], dtype=np.int32)
        self.roof_indices = np.zeros(cons.roof_grid.shape[:2], dtype=np.int32)
        self.obj_indices = np.zeros(len(data.obj_clr_idcs), dtype=np.int32)

        # Horizontal walls
        n_rows, n_cols = cons.hor_grid.shape[:-1]
        mask = data.hor_wall_mask

        for i in range(n_rows):
            for j in range(n_cols):
                pose = gymapi.Transform(gymapi.Vec3(
                    *cons.hor_grid[i, j],
                    cfg.WALL_HALFHEIGHT if mask[i, j] else cfg.WALL_HIDDEN_DEPTH))

                self.wally_handles[i, j] = handle = gym.create_actor(
                    self.handle,
                    sim.asset_wall_y,
                    pose,
                    f'wally-{i:02d}-{j:02d}',
                    self.group_id,
                    -1,
                    cfg.SEG_CLS_WALL)

                self.wally_indices[i, j] = gym.get_actor_index(self.handle, handle, gymapi.DOMAIN_SIM)

        self.wally_indices = self.wally_indices.flatten()

        # Vertical walls
        n_rows, n_cols = cons.ver_grid.shape[:-1]
        mask = data.ver_wall_mask

        for i in range(n_rows):
            for j in range(n_cols):
                pose = gymapi.Transform(gymapi.Vec3(
                    *cons.ver_grid[i, j],
                    cfg.WALL_HALFHEIGHT if mask[i, j] else cfg.WALL_HIDDEN_DEPTH))

                self.wallx_handles[i, j] = handle = gym.create_actor(
                    self.handle,
                    sim.asset_wall_x,
                    pose,
                    f'wallx-{i:02d}-{j:02d}',
                    self.group_id,
                    -1,
                    cfg.SEG_CLS_WALL)

                self.wallx_indices[i, j] = gym.get_actor_index(self.handle, handle, gymapi.DOMAIN_SIM)

        self.wallx_indices = self.wallx_indices.flatten()

        # Linking pillars
        n_rows, n_cols = cons.link_grid.shape[:-1]
        mask = data.sqr_link_mask

        for i in range(n_rows):
            for j in range(n_cols):
                pose = gymapi.Transform(gymapi.Vec3(
                    *cons.link_grid[i, j],
                    cfg.WALL_HALFHEIGHT if mask[i, j] else cfg.WALL_HIDDEN_DEPTH))

                self.link_handles[i, j] = handle = gym.create_actor(
                    self.handle,
                    sim.asset_link,
                    pose,
                    f'link-{i:02d}-{j:02d}',
                    self.group_id,
                    -1,
                    cfg.SEG_CLS_WALL)

                self.link_indices[i, j] = gym.get_actor_index(self.handle, handle, gymapi.DOMAIN_SIM)

        self.link_indices = self.link_indices.flatten()

        # Square roofs
        n_rows, n_cols = cons.roof_grid.shape[:-1]
        mask = data.sqr_roof_mask

        for i in range(n_rows):
            for j in range(n_cols):
                pose = gymapi.Transform(gymapi.Vec3(
                    *cons.roof_grid[i, j],
                    cfg.ROOF_HEIGHT if mask[i, j] else cfg.ROOF_HIDDEN_DEPTH))

                self.roof_handles[i, j] = handle = gym.create_actor(
                    self.handle,
                    sim.asset_roof,
                    pose,
                    f'roof-{i:02d}-{j:02d}',
                    self.group_id,
                    -1,
                    cfg.SEG_CLS_WALL)

                self.roof_indices[i, j] = gym.get_actor_index(self.handle, handle, gymapi.DOMAIN_SIM)

        self.roof_indices = self.roof_indices.flatten()

        # Objectives
        for i, pos in enumerate(data.obj_points):
            pose = gymapi.Transform(gymapi.Vec3(*pos, cfg.OBJECT_HEIGHT))

            self.obj_handles[i] = handle = gym.create_actor(
                self.handle,
                sim.asset_object,
                pose,
                f'object-{i:02d}',
                self.group_id,
                -1,
                cfg.SEG_CLS_OBJ)

            self.obj_indices[i] = gym.get_actor_index(self.handle, handle, gymapi.DOMAIN_SIM)

    def init_agents(self):
        """Place agents with DOF and camera properties."""

        sim = self.sim
        gym = sim.gym
        data = self.data

        self.bot_handles = np.zeros(sim.n_bots, dtype=np.int32)
        self.bot_indices = np.zeros(sim.n_bots, dtype=np.int32)

        # Bots
        for i, pos, angle in zip(range(sim.n_bots), data.bot_spawn_points, data.bot_spawn_angles):
            pose = gymapi.Transform(
                gymapi.Vec3(*pos, 0.),
                gymapi.Quat.from_euler_zyx(0., 0., angle))

            self.bot_handles[i] = handle = gym.create_actor(
                    self.handle,
                    sim.asset_bot,
                    pose,
                    f'bot-{i:02d}',
                    self.group_id,
                    -1,
                    cfg.SEG_CLS_BOT)

            self.bot_indices[i] = gym.get_actor_index(self.handle, handle, gymapi.DOMAIN_SIM)

            gym.set_rigid_body_segmentation_id(self.handle, handle, BOT_BODY_IDX, cfg.SEG_CLS_BODY)
            gym.set_rigid_body_segmentation_id(self.handle, handle, BOT_CARGO_IDX, cfg.SEG_CLS_CARGO)

        # Set bot motor params
        dof_props: 'dict[str, ndarray]' = gym.get_actor_dof_properties(self.handle, handle)
        dof_props['driveMode'].fill(gymapi.DOF_MODE_EFFORT)
        dof_props['stiffness'].fill(MOT_STIFFNESS)
        dof_props['damping'].fill(MOT_DAMPING)

        for handle in self.bot_handles:
            gym.set_actor_dof_properties(self.handle, handle, dof_props)

        # Init visual sensors
        camera_props = gymapi.CameraProperties()
        camera_props.enable_tensors = True
        camera_props.height = cfg.OBS_IMG_RES_HEIGHT
        camera_props.width = cfg.OBS_IMG_RES_WIDTH
        # camera_props.horizontal_fov = 90.     # Default
        camera_props.far_plane = 128.           # 2e+6 by default
        camera_pos = gymapi.Vec3(*CAM_OFFSET)

        self.cam_handles = np.zeros_like(self.bot_handles)

        for i, bot_handle in enumerate(self.bot_handles):
            self.cam_handles[i] = cam_handle = gym.create_camera_sensor(self.handle, camera_props)
            body_handle = gym.get_actor_rigid_body_handle(self.handle, bot_handle, BOT_BODY_IDX)

            gym.attach_camera_to_body(
                cam_handle, self.handle, body_handle, gymapi.Transform(camera_pos), gymapi.FOLLOW_TRANSFORM)

    def set_colours(self, full: bool = True):
        sim = self.sim
        gym = sim.gym
        data = self.data

        # Objects
        object_colours = self.OBJECT_COLOURS[data.obj_clr_idcs]

        for colour, handle in zip(object_colours, self.obj_handles):
            gym.set_rigid_body_color(self.handle, handle, OBJECT_BODY_IDX, gymapi.MESH_VISUAL, colour)

        # Bots
        for handle in self.bot_handles:
            gym.set_rigid_body_color(self.handle, handle, BOT_BODY_IDX, gymapi.MESH_VISUAL, self.BASE_BOT_COLOUR)
            gym.set_rigid_body_color(self.handle, handle, BOT_CARGO_IDX, gymapi.MESH_VISUAL, self.BASE_CARGO_COLOUR)

        if not full or sim.is_preset:
            return

        # Walls
        for i in range(self.wally_handles.shape[0]):
            for j in range(self.wally_handles.shape[1]):
                handle = self.wally_handles[i, j]
                colour = self.WALL_COLOURS[data.hor_wall_clr_idcs[i, j]]

                gym.set_rigid_body_color(self.handle, handle, OBJECT_BODY_IDX, gymapi.MESH_VISUAL, colour)

        for i in range(self.wallx_handles.shape[0]):
            for j in range(self.wallx_handles.shape[1]):
                handle = self.wallx_handles[i, j]
                colour = self.WALL_COLOURS[data.ver_wall_clr_idcs[i, j]]

                gym.set_rigid_body_color(self.handle, handle, OBJECT_BODY_IDX, gymapi.MESH_VISUAL, colour)

        # Links
        for handle in self.link_handles.flatten():
            gym.set_rigid_body_color(self.handle, handle, OBJECT_BODY_IDX, gymapi.MESH_VISUAL, self.LINK_COLOUR)

        # Roofs
        for i in range(self.roof_handles.shape[0]):
            for j in range(self.roof_handles.shape[1]):
                handle = self.roof_handles[i, j]
                colour = self.WALL_COLOURS[data.sqr_roof_clr_idcs[i, j]]

                gym.set_rigid_body_color(self.handle, handle, OBJECT_BODY_IDX, gymapi.MESH_VISUAL, colour)

    def set_rigid_props(self):
        if not self.sim.is_preset:
            return

        gym = self.sim.gym

        for handle in self.bot_handles:
            shape_prop_list = gym.get_actor_rigid_shape_properties(self.handle, handle)

            for shape_props in shape_prop_list:
                shape_props.friction = 0.
                # shape_props.rolling_friction
                # shape_props.torsion_friction

            gym.set_actor_rigid_shape_properties(self.handle, handle, shape_prop_list)

    def reset(self, full: bool = True) -> ndarray:
        """Partially or fully reset the environment and relay the new states."""

        sim = self.sim
        cons = sim.constructor

        self.data = data = cons.generate(None if full else self.data, not sim.is_preset)
        self.set_colours(full)
        self.set_rigid_props()

        if full and not sim.is_preset:
            wally_states = np.concatenate((
                cons.hor_grid.reshape(-1, 2),
                np.where(
                    data.hor_wall_mask.flatten(),
                    cfg.WALL_HALFHEIGHT,
                    cfg.WALL_HIDDEN_DEPTH).astype(np.float32)[:, None],
                sim.wall_substate), axis=-1, dtype=np.float32)

            wallx_states = np.concatenate((
                cons.ver_grid.reshape(-1, 2),
                np.where(
                    data.ver_wall_mask.flatten(),
                    cfg.WALL_HALFHEIGHT,
                    cfg.WALL_HIDDEN_DEPTH).astype(np.float32)[:, None],
                sim.wall_substate), axis=-1, dtype=np.float32)

            link_states = np.concatenate((
                cons.link_grid.reshape(-1, 2),
                np.where(
                    data.sqr_link_mask.flatten(),
                    cfg.WALL_HALFHEIGHT,
                    cfg.WALL_HIDDEN_DEPTH).astype(np.float32)[:, None],
                sim.link_substate), axis=-1, dtype=np.float32)

            roof_states = np.concatenate((
                cons.roof_grid.reshape(-1, 2),
                np.where(
                    data.sqr_roof_mask.flatten(),
                    cfg.ROOF_HEIGHT,
                    cfg.ROOF_HIDDEN_DEPTH).astype(np.float32)[:, None],
                sim.roof_substate), axis=-1, dtype=np.float32)

            static_states = (wally_states, wallx_states, link_states, roof_states)

        else:
            static_states = ()

        obj_states = np.concatenate((
            data.obj_points,
            sim.obj_substate), axis=-1, dtype=np.float32)

        bot_states = np.concatenate((
            data.bot_spawn_points,
            self.sim.bot_substates[0],
            Rotation.from_euler('z', data.bot_spawn_angles).as_quat(),
            self.sim.bot_substates[1]), axis=-1, dtype=np.float32)

        actor_states = np.concatenate((*static_states, obj_states, bot_states))

        return actor_states


class MazeSim:
    """
    Gym.Sim wrapper for procedurally generated mazes.

    NOTE: The main bottleneck is the number/overhead of cameras and objects for
    them to render. Camera resolution and physics simulation have less effect.
    """

    DEFAULT_SIM_ARGS = Namespace(
        use_gpu_pipeline=True,
        use_gpu=True,
        num_threads=0,
        compute_device_id=0,
        graphics_device_id=0)

    env_width: int
    n_grid_segments: int
    n_graph_points: int
    n_bots: int
    n_objects: int
    ep_duration: int
    rng_seed: int

    def __init__(
        self,
        level: int = 4,
        n_bots: int = 0,
        n_envs: int = 0,
        n_objects: int = 0,
        n_near_bots: int = None,
        n_obj_colours: int = None,
        mandated_clr_indices: 'tuple[int, ...]' = None,
        fps: int = 60,
        args: Namespace = DEFAULT_SIM_ARGS,
        rng: 'None | np.random.Generator' = None,
        preset_path: str = None
    ):
        self.is_preset = preset_path is not None

        # Init IsaacGym
        self.gym = gym = gymapi.acquire_gym()

        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0., 0., -9.80665)
        sim_params.dt = 1. / fps
        sim_params.substeps = max(128 // fps, 1)

        # NOTE: Params behave differently between CPU and GPU
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

        if args.use_gpu_pipeline:
            if args.graphics_device_id != args.compute_device_id:
                print(f'Warning: Overriding graphics device {args.graphics_device_id} to {args.compute_device_id}.')

                args.graphics_device_id = args.compute_device_id

        self.handle = gym.create_sim(args.compute_device_id, args.graphics_device_id, gymapi.SIM_PHYSX, sim_params)

        if self.handle is None:
            raise Exception('Failed to initialise.')

        # Add ground plane
        # NOTE: Slipping on CPU
        # NOTE: Height field has uniform colour
        if not self.is_preset:
            plane_params = gymapi.HeightFieldParams()
            plane_params.dynamic_friction = 0.                      # 1. by default
            plane_params.segmentation_id = cfg.SEG_CLS_PLANE
            plane_params.column_scale = 500
            plane_params.row_scale = 500
            plane_params.nbColumns = 2
            plane_params.nbRows = 2
            plane_params.transform = gymapi.Transform(gymapi.Vec3(-65., -50., 0.))
            height_samples = np.array((0, 0, 0, 0), dtype=np.int16)
            gym.add_heightfield(self.handle, height_samples, plane_params)

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

        # Assign level params
        self.level = level

        for key, val in cfg.LEVEL_PARAMS[level].items():
            setattr(self, key, val)

        self.n_objects = n_objects if n_objects > 0 else self.n_objects
        self.n_bots = n_bots if n_bots > 0 else self.n_bots
        self.n_envs = n_envs if n_envs > 0 else 1
        self.n_envs_per_row: int = round(self.n_envs**0.5)
        self.n_all_bots: int = self.n_envs * self.n_bots

        # Parallel environments are created with the same base parameters
        if not self.is_preset:
            preset_data_dict = None
            supenv_width = None
            n_supgrid_segments = None
            preset_assets = None
            self.preset_specs = None

        else:
            preset_data_dict = np.load(preset_path + '.npz')
            supgrid_delims = preset_data_dict['grid_delims']
            supenv_width = supgrid_delims[-1] - supgrid_delims[0]
            n_supgrid_segments = len(supgrid_delims) - 1

            with open(preset_path + '.json', 'r') as spec_file:
                preset_cfg_dict = json.load(spec_file)

            preset_assets = preset_cfg_dict['assets']
            self.preset_specs = preset_cfg_dict['specs']

        self.constructor = MazeConstructor(
            self.env_width,
            self.n_grid_segments,
            self.n_graph_points,
            self.n_bots,
            self.n_objects,
            n_near_bots,
            n_obj_colours,
            mandated_clr_indices,
            self.rng_seed if rng is None else rng,
            supenv_width,
            n_supgrid_segments)

        self.open_grid_delims = self.constructor.open_grid_delims

        if preset_data_dict is None:
            env_data = [self.constructor.generate() for _ in range(self.n_envs)]

        else:
            preset_data = MazeData(self.constructor, precompute=False, **preset_data_dict)

            env_data = [self.constructor.generate(preset_data.copy()) for _ in range(self.n_envs)]

        # Load bot and objective assets
        bot_asset_options = gymapi.AssetOptions()
        bot_asset_options.thickness = 0.005  # 0.02 default; change not visible, but this relates better to bot size

        self.asset_bot = gym.load_urdf(self.handle, cfg.ASSET_DIR, BOT_FILE_NAME, bot_asset_options)
        self.asset_object = gym.load_urdf(self.handle, cfg.ASSET_DIR, OBJ_FILE_NAME, bot_asset_options)

        # Create static assets
        static_asset_options = gymapi.AssetOptions()
        static_asset_options.fix_base_link = True
        wall_length = self.env_width / self.n_grid_segments

        self.asset_wall_x = self.gym.create_box(
            self.handle, wall_length-cfg.WALL_WIDTH, cfg.WALL_WIDTH, cfg.WALL_HEIGHT, static_asset_options)

        self.asset_wall_y = self.gym.create_box(
            self.handle, cfg.WALL_WIDTH, wall_length-cfg.WALL_WIDTH, cfg.WALL_HEIGHT, static_asset_options)

        self.asset_link = gym.create_box(
            self.handle, cfg.WALL_WIDTH, cfg.WALL_WIDTH, cfg.WALL_HEIGHT, static_asset_options)

        self.asset_roof = gym.create_box(
            self.handle, wall_length-cfg.WALL_WIDTH, wall_length-cfg.WALL_WIDTH, cfg.WALL_WIDTH, static_asset_options)

        # Create preset assets
        if preset_assets is None:
            self.preset_assets = None

        else:
            self.preset_assets = {}

            for key, vals in preset_assets.items():
                width, length, height = vals

                self.preset_assets[key] = (
                    gym.create_box(self.handle, width, length, height, static_asset_options),
                    height / 2.)

        # Build envs from generated and loaded data
        self.envs = [MazeEnv(self, maze_data, group_id) for group_id, maze_data in enumerate(env_data)]

        # Gather env data
        self.all_wallgrid_pairs = np.stack([env.data.grid_wall_pairs for env in self.envs])

        self.env_bot_handles = np.array([
            (env.handle, bot_handle)
            for env in self.envs
            for bot_handle in env.bot_handles], dtype=object)

        # Data for env resetting
        n_wallx = np.prod(self.constructor.ver_grid.shape[:-1])
        n_links = np.prod(self.constructor.link_grid.shape[:-1])
        n_roofs = np.prod(self.constructor.roof_grid.shape[:-1])

        self.wall_substate = np.concatenate((
            np.zeros((n_wallx, 3), dtype=np.float32),
            np.ones((n_wallx, 1), dtype=np.float32),
            np.zeros((n_wallx, 6), dtype=np.float32)), axis=-1, dtype=np.float32)

        self.link_substate = np.concatenate((
            np.zeros((n_links, 3), dtype=np.float32),
            np.ones((n_links, 1), dtype=np.float32),
            np.zeros((n_links, 6), dtype=np.float32)), axis=-1, dtype=np.float32)

        self.roof_substate = np.concatenate((
            np.zeros((n_roofs, 3), dtype=np.float32),
            np.ones((n_roofs, 1), dtype=np.float32),
            np.zeros((n_roofs, 6), dtype=np.float32)), axis=-1, dtype=np.float32)

        self.obj_substate = np.concatenate((
            np.ones((self.n_objects, 1), dtype=np.float32) * cfg.OBJECT_HEIGHT,
            np.zeros((self.n_objects, 3), dtype=np.float32),
            np.ones((self.n_objects, 1), dtype=np.float32),
            np.zeros((self.n_objects, 6), dtype=np.float32)), axis=-1, dtype=np.float32)

        self.bot_substates = (
            np.zeros((self.n_bots, 1), dtype=np.float32),
            np.zeros((self.n_bots, 6), dtype=np.float32))

    def reset(self, rst_env_indices: 'list[int]', full: bool = True) -> 'tuple[ndarray, ndarray]':
        actor_states = [self.envs[i].reset(full) for i in rst_env_indices]

        if full and not self.is_preset:
            for i in rst_env_indices:
                self.all_wallgrid_pairs[i] = self.envs[i].data.grid_wall_pairs

            env_idx_keys = (
                'wally_indices', 'wallx_indices', 'link_indices', 'roof_indices', 'obj_indices', 'bot_indices')

        else:
            env_idx_keys = ('obj_indices', 'bot_indices')

        actor_indices = [getattr(self.envs[i], k) for i in rst_env_indices for k in env_idx_keys]

        return np.concatenate(actor_states), np.concatenate(actor_indices)

    def cleanup(self):
        """Destroy existing envs and sim, along with their data."""

        for env in self.envs:
            for cam_handle in env.cam_handles:
                self.gym.destroy_camera_sensor(self.handle, env.handle, cam_handle)

            self.gym.destroy_env(env.handle)

        self.envs.clear()
        self.gym.destroy_sim(self.handle)
