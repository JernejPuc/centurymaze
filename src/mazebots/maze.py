"""Procedural generation of maze-like environments"""

import os
from argparse import ArgumentParser, Namespace
from collections import deque
from itertools import chain

import numpy as np
from numpy import ndarray
from scipy.cluster.vq import kmeans2
from scipy.spatial.transform import Rotation
from isaacgym import gymapi

import config as cfg
from utils import (
    any_intersections,
    astar,
    eval_path,
    get_numba_dict,
    min_distance,
    prune_path_backward,
    prune_path_forward,
    reconstruct_paths,
    urquhart)


# Base parameters for procedural environment generation
# At higher levels, mazes become larger and more complex,
# with more agents and objectives and longer episodes
LEVEL_PARAMS = {
    1: {
        'env_width': 4,
        'n_grid_segments': 1,
        'n_graph_points': 6,
        'n_bots': 2,
        'n_objects': 2,
        'ep_duration': 8,
        'rng_seed': 12},
    2: {
        'env_width': 6,
        'n_grid_segments': 3,
        'n_graph_points': 9,
        'n_bots': 4,
        'n_objects': 3,
        'ep_duration': 32,
        'rng_seed': 42},
    3: {
        'env_width': 9,
        'n_grid_segments': 5,
        'n_graph_points': 16,
        'n_bots': 8,
        'n_objects': 4,
        'ep_duration': 80,
        'rng_seed': 5927},
    4: {
        'env_width': 12,
        'n_grid_segments': 7,
        'n_graph_points': 30,
        'n_bots': 16,
        'n_objects': 5,
        'ep_duration': 140,
        'rng_seed': 42},
    5: {
        'env_width': 18,
        'n_grid_segments': 10,
        'n_graph_points': 56,
        'n_bots': 32,
        'n_objects': 6,
        'ep_duration': 240,
        'rng_seed': 50},
    6: {
        'env_width': 24,
        'n_grid_segments': 14,
        'n_graph_points': 92,
        'n_bots': 64,
        'n_objects': 7,
        'ep_duration': 360,
        'rng_seed': 1},
    7: {
        'env_width': 34,
        'n_grid_segments': 20,
        'n_graph_points': 160,
        'n_bots': 128,
        'n_objects': 8,
        'ep_duration': 600,
        'rng_seed': 64578}}

# Colour palette
COLOURS = {
    'pastel': np.array([
        [185, 255, 175],
        [255, 255, 160],
        [255, 160, 205],
        [255, 205, 135],
        [175, 195, 255],
        [225, 160, 255],
        [175, 245, 255],
        [125, 235, 205],
        [255, 215, 215]]) / 255.,
    'basic': np.array([
        [155, 205, 0],
        [0, 145, 0],
        [255, 145, 0],
        [0, 0, 255],
        [0, 160, 200],
        [255, 0, 0],
        [100, 50, 0],
        [115, 20, 165],
        [215, 20, 135]]) / 255.,
    'grey': np.array([
        [0, 0, 0],
        [51, 51, 51],
        [255, 255, 255]]) / 255.}

# Assets
BOT_FILE_DIR = os.path.abspath(os.path.join(__file__, '..', 'assets'))
BOT_FILE_NAME = 'mazebot.urdf'

OBJECT_BODY_IDX = 0
BOT_BODY_IDX = 0
BOT_CARGO_IDX = 1

BOT_WIDTH = 0.3
BOT_RADIUS = BOT_WIDTH/2 * 2**0.5
BOT_HEIGHT = 0.264

# Lowered inide the body, because the cargo was clipping in any configuration
CAM_OFFSET = np.array([BOT_WIDTH/2. - 0.025, 0., BOT_HEIGHT - 0.1])

OBJECT_RADIUS = 0.225
OBJECT_HEIGHT = 0.275

WALL_WIDTH = 0.05
WALL_HEIGHT = 0.75
WALL_HALFWIDTH = WALL_WIDTH/2
WALL_HALFHEIGHT = WALL_HEIGHT/2

# Some offset is added to prevent accidental collisions on startup
MIN_BUFFER = 0.025
WALL_HIDDEN_DEPTH = -(WALL_HALFHEIGHT + MIN_BUFFER)

OBJECT_TO_WALL_BUFFER = OBJECT_RADIUS + WALL_HALFWIDTH + MIN_BUFFER
BOT_TO_BOT_BUFFER = BOT_RADIUS*2 + MIN_BUFFER
BOT_TO_OBJECT_BUFFER = OBJECT_RADIUS + BOT_RADIUS + MIN_BUFFER
BOT_TO_WALL_BUFFER = BOT_RADIUS + WALL_HALFWIDTH + MIN_BUFFER

GOAL_RADIUS = 1.

MOT_STIFFNESS = 0.001   # Bot won't move at 0.1, slow turning difficult at 0.01
MOT_DAMPING = 0.02      # Applying torque without damping makes the bot fly off
MOT_MAX_TORQUE = 1.     # Per motor/wheel


class MazeData:
    """Describes the topology and functional elements of a specific environment."""

    urq_graph_points: ndarray
    urq_graph_edges: ndarray

    grid_conn_map: 'dict[int, ndarray[int]]'
    path_map: 'dict[tuple[int, int], ndarray[int]]'

    open_grid_delims: ndarray
    grid_square_centres: ndarray
    grid_wall_pairs: ndarray
    grid_hor_wall_mask: ndarray
    grid_ver_wall_mask: ndarray
    grid_link_mask: ndarray

    obj_points: ndarray
    obj_trans_probs: ndarray
    obj_clr_idcs: ndarray
    wall_clr_idcs: ndarray

    bot_spawn_points: ndarray
    bot_spawn_angles: ndarray

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

        self.init_connection_map()
        self.init_path_map()

    def init_connection_map(self):
        """Build a graph of connections between connected squares on the grid."""

        n_segments = len(self.open_grid_delims) + 1

        hor_con_mask = ~self.grid_hor_wall_mask
        ver_con_mask = ~self.grid_ver_wall_mask

        connection_graph: 'dict[int, list[int]]' = {
            (i*n_segments+j): []
            for i in range(n_segments)
            for j in range(n_segments)}

        # Iterate over grid square index coordinates
        for i, i_prev, i_post in zip(range(n_segments), range(-1, n_segments-1), range(1, n_segments+1)):
            f_row_idx = i * n_segments
            f_prev_idx = i_prev * n_segments
            f_post_idx = i_post * n_segments

            for j, j_post in zip(range(n_segments), range(1, n_segments+1)):
                og_node = (i, j)
                hor_node = (i, j_post)
                ver_node = (i_post, j)
                ldiag_node = (i_post, j_post)
                udiag_node = (i_prev, j_post)

                # Check for connections
                hor_con = (j_post < n_segments) and ver_con_mask[hor_node]
                ver_con = (i_post < n_segments) and hor_con_mask[ver_node]

                ldiag_con = (
                    hor_con and
                    ver_con and
                    hor_con_mask[ldiag_node] and
                    ver_con_mask[ldiag_node])

                udiag_con = (
                    hor_con and
                    hor_con_mask[og_node] and
                    ver_con_mask[udiag_node] and
                    hor_con_mask[hor_node])

                # Transform index coordinates to indices of a flattened array
                og_node = f_row_idx + j
                hor_node = f_row_idx + j_post
                ver_node = f_post_idx + j
                ldiag_node = f_post_idx + j_post
                udiag_node = f_prev_idx + j_post

                # Add connections to the graph
                if hor_con:
                    connection_graph[og_node].append(hor_node)
                    connection_graph[hor_node].append(og_node)

                if ver_con:
                    connection_graph[og_node].append(ver_node)
                    connection_graph[ver_node].append(og_node)

                if ldiag_con:
                    connection_graph[og_node].append(ldiag_node)
                    connection_graph[ldiag_node].append(og_node)

                if udiag_con:
                    connection_graph[og_node].append(udiag_node)
                    connection_graph[udiag_node].append(og_node)

        # Convert connection graph into numba typed dict
        self.grid_conn_map = get_numba_dict()

        for key, lst in connection_graph.items():
            self.grid_conn_map[key] = np.array(lst, dtype=np.int64)

    def init_path_map(self):
        """
        Use A* on the underlying graph to precompute estimated paths
        from all valid entry points to all target objects.
        """

        self.path_map = get_numba_dict(tuple_as_key=True)

        n_segments = len(self.open_grid_delims) + 1
        exit_idcs = np.digitize(self.obj_points, self.open_grid_delims)

        flattened_exit_idcs = n_segments * exit_idcs[:, 0] + exit_idcs[:, 1]

        for entry_node, v in self.grid_conn_map.items():
            if len(v) == 0:
                continue

            for i, exit_node in enumerate(flattened_exit_idcs):
                path = astar(self.grid_conn_map, self.grid_square_centres, entry_node, exit_node)

                self.path_map[entry_node, exit_node] = prune_path_backward(
                    path,
                    exit_idcs[i],
                    self.obj_points[i],
                    self.grid_square_centres,
                    self.grid_wall_pairs,
                    n_segments)

    def get_path_estimate(
        self,
        start_pts: ndarray,
        end_pts: ndarray,
        sight_mask: ndarray
    ) -> 'tuple[ndarray, ndarray]':
        """
        Estimate path length and starting direction from valid starting points
        to target end points based on precomputed A* reference paths.
        """

        return reconstruct_paths(
            start_pts,
            end_pts,
            sight_mask,
            self.path_map,
            self.open_grid_delims,
            self.grid_square_centres,
            self.grid_wall_pairs)

    def reconstruct_path(self, start_pt: ndarray, end_pt: ndarray) -> 'tuple[ndarray, float, ndarray]':
        """Estimate path length and starting direction for a single test case."""

        n_segments = len(self.open_grid_delims) + 1

        entry_idcs = np.digitize(start_pt, self.open_grid_delims)
        exit_idcs = np.digitize(end_pt, self.open_grid_delims)

        entry_node = n_segments * entry_idcs[0] + entry_idcs[1]
        exit_node = n_segments * exit_idcs[0] + exit_idcs[1]

        path = prune_path_forward(
            self.path_map[entry_node, exit_node],
            entry_idcs,
            start_pt,
            self.grid_square_centres,
            self.grid_wall_pairs,
            n_segments)

        path_pts = self.grid_square_centres[path]

        dist, dir_ = eval_path(path, start_pt, end_pt, self.grid_square_centres)

        return path_pts, dist, dir_


class MazeConstructor:
    """Processes base parameters to generate random environments."""

    MAX_OBJECT_SHUFFLES = 7
    MAX_BOT_SHUFFLES = 2

    def __init__(
        self,
        env_width: int,
        n_grid_segments: int,
        n_graph_points: int,
        n_bots: int,
        n_objects: int,
        rng: 'None | int | np.random.Generator' = None
    ):
        self.env_halfwidth = env_width / 2.
        self.min_object_spacing = max(env_width / 3., GOAL_RADIUS*2 + MIN_BUFFER)

        # Grid delimiters
        self.grid_delims = np.linspace(-self.env_halfwidth, self.env_halfwidth, n_grid_segments+1)
        self.open_grid_delims = self.grid_delims[1:-1]
        delim_centres = (self.grid_delims[1:] + self.grid_delims[:-1]) / 2.

        y_link, x_link = np.meshgrid(self.grid_delims, self.grid_delims)
        y_hor, x_hor = np.meshgrid(delim_centres, self.grid_delims)
        y_ver, x_ver = np.meshgrid(self.grid_delims, delim_centres)

        # Link and wall centres
        self.link_grid = np.stack((x_link, y_link), axis=-1)
        self.hor_grid = np.stack((x_hor, y_hor), axis=-1)
        self.ver_grid = np.stack((x_ver, y_ver), axis=-1)

        self.n_graph_points = n_graph_points
        self.n_bots = n_bots
        self.n_objects = n_objects

        self.rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

    def prune_grid(
        self,
        graph_edges: ndarray
    ) -> 'tuple[ndarray, ndarray, ndarray, ndarray, ndarray]':
        """Mask edges that intersect any edge of the given graph."""

        n_delims = len(self.grid_delims)
        n_segments = n_delims - 1

        # NOTE: [(0., 0.), (0., 0.)] is considered a null edge (clearing)
        grid_pairs_masked = np.zeros((n_delims, n_delims, 2, 2, 2))

        grid_link_mask = np.zeros((n_delims, n_delims), dtype=np.bool8)
        grid_hor_mask = np.zeros((n_delims, n_segments), dtype=np.bool8)
        grid_ver_mask = np.zeros((n_segments, n_delims), dtype=np.bool8)

        remaining_edges = deque()

        # Iterate over pairs of edges in the grid
        for i in range(n_delims):
            for j in range(n_delims):

                # Check for intersections
                if j < n_segments:
                    hor_edge = self.link_grid[i, j:j+2]

                    if not any_intersections(*hor_edge, graph_edges):
                        grid_link_mask[i, j:j+2] = True
                        grid_hor_mask[i, j] = True
                        grid_pairs_masked[i, j, 0] = hor_edge
                        remaining_edges.append(hor_edge)

                if i < n_segments:
                    ver_edge = self.link_grid[i:i+2, j]

                    if not any_intersections(*ver_edge, graph_edges):
                        grid_link_mask[i:i+2, j] = True
                        grid_ver_mask[i, j] = True
                        grid_pairs_masked[i, j, 1] = ver_edge
                        remaining_edges.append(ver_edge)

        remaining_centres = np.array(remaining_edges).mean(axis=1)

        return remaining_centres, grid_pairs_masked, grid_link_mask, grid_hor_mask, grid_ver_mask

    def generate(self, data: 'None | MazeData' = None) -> MazeData:
        """Randomly generate mazes until the underlying graph is feasible."""

        while True:
            if data is None:
                # Generate graph vertices
                graph_points = self.rng.uniform(
                    low=-self.env_halfwidth,
                    high=self.env_halfwidth,
                    size=(self.n_graph_points, 2))

                # Construct a cyclical, connected graph
                graph_edges = urquhart(graph_points)

                # Create corridors where the graph's edges overlap with the grid
                (
                    remaining_centres,
                    grid_pairs_masked,
                    grid_link_mask,
                    grid_hor_mask,
                    grid_ver_mask) = self.prune_grid(graph_edges)

                # Spawn points must lie on the graph (vertices and points along the edges)
                candidate_points = np.concatenate((graph_points, np.mean(graph_edges, axis=1)))

            else:
                graph_points = data.urq_graph_points
                graph_edges = data.urq_graph_edges
                remaining_centres = None
                grid_pairs_masked = data.grid_wall_pairs
                grid_link_mask = data.grid_link_mask
                grid_hor_mask = data.grid_hor_wall_mask
                grid_ver_mask = data.grid_ver_wall_mask

                candidate_points = np.concatenate((graph_points, np.mean(graph_edges, axis=1)))
                self.rng.shuffle(candidate_points)
                self.rng.shuffle(graph_points)

            # Try to designate n_objects + n_bots points on the maze graph
            # as objectives and bot spawn points, respectively
            # Reconstruct the graph until all objects and spawn points are selected
            try:
                object_points = self.select_points(
                    graph_points,
                    self.n_objects, self.min_object_spacing,
                    None, 0.,
                    grid_pairs_masked, OBJECT_TO_WALL_BUFFER,
                    self.MAX_OBJECT_SHUFFLES)

                bot_spawn_points = self.select_points(
                    candidate_points,
                    self.n_bots, BOT_TO_BOT_BUFFER,
                    object_points, BOT_TO_OBJECT_BUFFER,
                    grid_pairs_masked, BOT_TO_WALL_BUFFER,
                    self.MAX_BOT_SHUFFLES)

                break

            except AssertionError:
                pass

        # Sample unique object colours
        object_colour_idx_order = self.rng.permutation(len(MazeEnv.OBJECT_COLOURS))
        object_colour_idcs = object_colour_idx_order[:self.n_objects]

        # Sample bot spawn angles uniformly
        bot_spawn_angles = self.rng.uniform(low=-np.pi, high=np.pi, size=self.n_bots)

        # Sample objective transition probabilities
        object_trans_probs = 0.8 + 0.2 * self.rng.uniform(size=(self.n_objects, self.n_objects))
        object_trans_probs[np.diag_indices_from(object_trans_probs)] = 0.
        object_trans_probs /= object_trans_probs.sum(axis=-1)[..., None]

        # Update existing maze
        if data is not None:
            data.obj_points = object_points
            data.obj_clr_idcs = object_colour_idcs
            data.obj_trans_probs = object_trans_probs

            data.bot_spawn_points = bot_spawn_points
            data.bot_spawn_angles = bot_spawn_angles

            data.init_path_map()

            return data

        # Sample wall colours
        wall_colour_idx_order = self.rng.permutation(len(MazeEnv.WALL_COLOURS))

        # Assign wall colours in clusters based on position
        wall_positions = np.vstack((self.hor_grid.reshape(-1, 2), self.ver_grid.reshape(-1, 2)))

        # NOTE: Empty clusters (not associated with any point) are auto-resolved with random points
        centroids = (
            remaining_centres
            if len(remaining_centres) <= len(MazeEnv.WALL_COLOURS)
            else kmeans2(remaining_centres, len(MazeEnv.WALL_COLOURS), seed=self.rng)[0])

        closest_centroid_idx = np.argmin(np.linalg.norm(wall_positions[None] - centroids[:, None], axis=-1), axis=0)
        wall_colour_idcs = wall_colour_idx_order[closest_centroid_idx]

        # Node positions for A*
        grid_square_centres = (self.grid_delims[1:] + self.grid_delims[:-1]) / 2.
        sqc_y, sqc_x = np.meshgrid(grid_square_centres, grid_square_centres)
        grid_square_centres = np.stack((sqc_x, sqc_y), axis=-1).reshape(-1, 2)

        return MazeData(
            urq_graph_points=graph_points,
            urq_graph_edges=graph_edges,
            open_grid_delims=self.open_grid_delims,
            grid_square_centres=grid_square_centres,
            grid_wall_pairs=grid_pairs_masked,
            grid_link_mask=grid_link_mask,
            grid_hor_wall_mask=grid_hor_mask,
            grid_ver_wall_mask=grid_ver_mask,
            obj_points=object_points,
            obj_trans_probs=object_trans_probs,
            obj_clr_idcs=object_colour_idcs,
            wall_clr_idcs=wall_colour_idcs,
            bot_spawn_points=bot_spawn_points,
            bot_spawn_angles=bot_spawn_angles)

    def select_points(
        self,
        candidate_points: ndarray,
        n_to_select: int,
        min_dist_to_same_pts: float = 1.,
        other_points: ndarray = None,
        min_dist_to_other_pts: float = 1.,
        grid_edges: ndarray = None,
        min_dist_to_grid_edges: float = 1.,
        max_shuffles: int = 1
    ) -> ndarray:
        """
        Greedy point selection with point and edge constraints.
        If an iteration does not find n_to_select suitable points,
        selection can be retried with the same points in a different order.
        """

        points = np.zeros((n_to_select, 2))

        for _ in range(max_shuffles):
            last_idx = 0

            for point in candidate_points:
                # Check distance to points of a different kind
                if (
                    (other_points is not None) and
                    np.linalg.norm(point - other_points, axis=1).min() < min_dist_to_other_pts
                ):
                    continue

                # Check distance to already selected points
                if (
                    last_idx > 0 and
                    np.linalg.norm(point - points[:last_idx], axis=1).min() < min_dist_to_same_pts
                ):
                    continue

                # Check distance to edges
                if grid_edges is not None:
                    sqr_idx, sqr_idy = np.digitize(point, self.open_grid_delims)
                    sqr_idx = max(1, min(len(self.open_grid_delims)-1, sqr_idx))
                    sqr_idy = max(1, min(len(self.open_grid_delims)-1, sqr_idy))

                    edge_subset = grid_edges[sqr_idx-1:sqr_idx+2, sqr_idy-1:sqr_idy+2].reshape(-1, 2, 2)

                    if any(min_distance(*edge, point) < min_dist_to_grid_edges for edge in edge_subset):
                        continue

                # Assign selected point
                points[last_idx] = point
                last_idx += 1

                if last_idx == n_to_select:
                    return points

            self.rng.shuffle(candidate_points)

        raise AssertionError


class MazeEnv:
    """
    Gym.Env wrapper for mazes inhabited by 4-wheeled robotic agents and static entities,
    i.e. walls and uniquely coloured objects.
    """

    ENV_HALFSPACING = 0.1

    WALL_COLOURS = np.array([gymapi.Vec3(*clr) for clr in COLOURS['pastel']], dtype=object)
    OBJECT_COLOURS = np.array([gymapi.Vec3(*clr) for clr in COLOURS['basic']], dtype=object)
    LINK_COLOUR = gymapi.Vec3(*COLOURS['grey'][1])
    BASE_BOT_COLOUR = gymapi.Vec3(*COLOURS['grey'][0])
    BASE_CARGO_COLOUR = gymapi.Vec3(*COLOURS['grey'][2])

    wallx_handles: ndarray
    wally_handles: ndarray
    link_handles: ndarray
    obj_handles: ndarray
    bot_handles: ndarray
    cam_handles: ndarray

    wallx_indices: ndarray
    wally_indices: ndarray
    link_indices: ndarray
    obj_indices: ndarray
    bot_indices: ndarray

    def __init__(self, sim: 'MazeSim', data: MazeData, group_id: int = 0):
        self.sim = sim
        self.data = data
        self.group_id = group_id

        # Place environment onto the grid
        env_border_half_size = sim.constructor.env_halfwidth + self.ENV_HALFSPACING

        env_bbox_vertex_low = gymapi.Vec3(-env_border_half_size, -env_border_half_size, 0.)
        env_bbox_vertex_high = gymapi.Vec3(env_border_half_size, env_border_half_size, 1.)

        self.handle = sim.gym.create_env(
            sim.handle,
            env_bbox_vertex_low,
            env_bbox_vertex_high,
            sim.n_envs_per_row)

        # Place entities (actors) into the environment
        self.init_static()
        self.init_agents()
        self.set_colours()
        self.set_rigid_props()

    def init_static(self):
        """Place objects and raise or lower walls and links wrt. associated masks."""

        sim = self.sim
        gym = sim.gym
        cons = sim.constructor
        data = self.data

        self.wally_handles = np.zeros(cons.hor_grid.shape[:2], dtype=np.int32)
        self.wallx_handles = np.zeros(cons.ver_grid.shape[:2], dtype=np.int32)
        self.link_handles = np.zeros(cons.link_grid.shape[:2], dtype=np.int32)
        self.obj_handles = np.zeros(len(data.obj_clr_idcs), dtype=np.int32)
        self.bot_handles = np.zeros(sim.n_bots, dtype=np.int32)

        self.wally_indices = np.zeros(cons.hor_grid.shape[:2], dtype=np.int32)
        self.wallx_indices = np.zeros(cons.ver_grid.shape[:2], dtype=np.int32)
        self.link_indices = np.zeros(cons.link_grid.shape[:2], dtype=np.int32)
        self.obj_indices = np.zeros(len(data.obj_clr_idcs), dtype=np.int32)
        self.bot_indices = np.zeros(sim.n_bots, dtype=np.int32)

        # Horizontal walls
        n_rows, n_cols = cons.hor_grid.shape[:-1]
        mask = data.grid_hor_wall_mask

        for i in range(n_rows):
            for j in range(n_cols):
                pose = gymapi.Transform(gymapi.Vec3(
                    *cons.hor_grid[i, j],
                    WALL_HALFHEIGHT if mask[i, j] else WALL_HIDDEN_DEPTH))

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
        mask = data.grid_ver_wall_mask

        for i in range(n_rows):
            for j in range(n_cols):
                pose = gymapi.Transform(gymapi.Vec3(
                    *cons.ver_grid[i, j],
                    WALL_HALFHEIGHT if mask[i, j] else WALL_HIDDEN_DEPTH))

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
        mask = data.grid_link_mask

        for i in range(n_rows):
            for j in range(n_cols):
                pose = gymapi.Transform(gymapi.Vec3(
                    *cons.link_grid[i, j],
                    WALL_HALFHEIGHT if mask[i, j] else WALL_HIDDEN_DEPTH))

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

        # Objectives
        for i, pos in enumerate(data.obj_points):
            pose = gymapi.Transform(gymapi.Vec3(*pos, OBJECT_HEIGHT))

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

        # Objects
        object_colours = self.OBJECT_COLOURS[self.data.obj_clr_idcs]

        for colour, handle in zip(object_colours, self.obj_handles):
            gym.set_rigid_body_color(self.handle, handle, OBJECT_BODY_IDX, gymapi.MESH_VISUAL, colour)

        # Bots
        for handle in self.bot_handles:
            gym.set_rigid_body_color(self.handle, handle, BOT_BODY_IDX, gymapi.MESH_VISUAL, self.BASE_BOT_COLOUR)
            gym.set_rigid_body_color(self.handle, handle, BOT_CARGO_IDX, gymapi.MESH_VISUAL, self.BASE_CARGO_COLOUR)

        if not full:
            return

        wall_colours = self.WALL_COLOURS[self.data.wall_clr_idcs]
        wall_handles = chain(self.wally_handles.flatten(), self.wallx_handles.flatten())

        # Walls
        for colour, handle in zip(wall_colours, wall_handles):
            gym.set_rigid_body_color(self.handle, handle, OBJECT_BODY_IDX, gymapi.MESH_VISUAL, colour)

        # Links
        for handle in self.link_handles.flatten():
            gym.set_rigid_body_color(self.handle, handle, OBJECT_BODY_IDX, gymapi.MESH_VISUAL, self.LINK_COLOUR)

    def set_rigid_props(self):
        # TODO: Test when/if friction var becomes relevant
        return

        gym = self.sim.gym
        rng = self.sim.constructor.rng

        for handle in self.bot_handles:
            rigid_props = gym.get_actor_rigid_shape_properties(self.handle, handle)
            (
                rigid_props.friction,
                rigid_props.rolling_friction,
                rigid_props.torsion_friction) = rng.uniform(-0.9, 1.1, 3)

            gym.set_actor_rigid_shape_properties(self.handle, handle, rigid_props)

    def reset(self, full: bool = True) -> ndarray:
        """Partially or fully reset the environment and relay the new states."""

        sim = self.sim
        cons = sim.constructor

        self.data = data = cons.generate(None if full else self.data)
        self.set_colours(full)
        self.set_rigid_props()

        if full:
            wally_states = np.concatenate((
                cons.hor_grid.reshape(-1, 2),
                np.where(
                    data.grid_hor_wall_mask.flatten(),
                    WALL_HALFHEIGHT,
                    WALL_HIDDEN_DEPTH).astype(np.float32)[:, None],
                sim.wall_substate), axis=-1, dtype=np.float32)

            wallx_states = np.concatenate((
                cons.ver_grid.reshape(-1, 2),
                np.where(
                    data.grid_ver_wall_mask.flatten(),
                    WALL_HALFHEIGHT,
                    WALL_HIDDEN_DEPTH).astype(np.float32)[:, None],
                sim.wall_substate), axis=-1, dtype=np.float32)

            link_states = np.concatenate((
                cons.link_grid.reshape(-1, 2),
                np.where(
                    data.grid_link_mask.flatten(),
                    WALL_HALFHEIGHT,
                    WALL_HIDDEN_DEPTH).astype(np.float32)[:, None],
                sim.link_substate), axis=-1, dtype=np.float32)

            states = (wally_states, wallx_states, link_states)

        else:
            states = ()

        obj_states = np.concatenate((
            data.obj_points,
            sim.obj_substate), axis=-1, dtype=np.float32)

        bot_states = np.concatenate((
            data.bot_spawn_points,
            self.sim.bot_substates[0],
            Rotation.from_euler('z', data.bot_spawn_angles).as_quat(),
            self.sim.bot_substates[1]), axis=-1, dtype=np.float32)

        actor_states = np.concatenate(states + (obj_states, bot_states))

        return actor_states


class MazeSim:
    """Gym.Sim wrapper for procedurally generated mazes."""

    # Benchmarks suggest highest total FPS at 256 agents/cameras with 3.5GB VRAM
    # The main bottleneck is the number/overhead of cameras
    # Camera resolution and physics simulation have less effect
    NUM_ALL_BOTS = 256

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
        fps: int = 60,
        args: Namespace = DEFAULT_SIM_ARGS,
        rng: 'None | np.random.Generator' = None
    ):
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
        sim_params.physx.contact_collection = gymapi.CC_NEVER   # All substeps by default; no tracking needed here

        self.handle = gym.create_sim(args.compute_device_id, args.graphics_device_id, gymapi.SIM_PHYSX, sim_params)

        if self.handle is None:
            raise Exception('Failed to initialise.')

        # Add ground plane
        # NOTE: Slipping on CPU
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        plane_params.dynamic_friction = 0.                      # 1. by default
        plane_params.segmentation_id = cfg.SEG_CLS_PLANE
        gym.add_ground(self.handle, plane_params)

        # Override lighting to get rid of some weird light from below
        # 0, (0.5, 0.5, 0.5), (0., 0., 0.), (1., 1., 1.)
        # 1, (0.5, 0.5, 0.5), (0., 0., 0.), (1., -1., 1.)
        # 2, (0.2, 0.2, 0.2), (0., 0., 0.), (-1., 1., -1.)
        # 3, (0., 0., 0.), (0., 0., 0.), (0., 0., 0.)
        gym.set_light_parameters(
            self.handle, 2, gymapi.Vec3(0.6, 0.6, 0.6), gymapi.Vec3(0., 0., 0.), gymapi.Vec3(-1., 0.33, 1.))

        # Assign level params
        self.level = level

        for key, val in LEVEL_PARAMS[level].items():
            setattr(self, key, val)

        self.n_bots = n_bots if n_bots > 0 else self.n_bots
        self.n_envs = n_envs if n_envs > 0 else (self.NUM_ALL_BOTS // self.n_bots)
        self.n_envs_per_row: int = round(self.n_envs**0.5)
        self.n_all_bots: int = self.n_envs * self.n_bots

        # Parallel environments are created with the same base parameters
        self.constructor = MazeConstructor(
            self.env_width,
            self.n_grid_segments,
            self.n_graph_points,
            self.n_bots,
            self.n_objects,
            self.rng_seed if rng is None else rng)

        self.open_grid_delims = self.constructor.open_grid_delims
        maze_data = [self.constructor.generate() for _ in range(self.n_envs)]

        # Load bot asset
        bot_asset_options = gymapi.AssetOptions()
        bot_asset_options.thickness = 0.005  # 0.02 default; change not visible, but this relates better to bot size

        self.asset_bot = gym.load_urdf(self.handle, BOT_FILE_DIR, BOT_FILE_NAME, bot_asset_options)

        # Create static assets
        static_asset_options = gymapi.AssetOptions()
        static_asset_options.fix_base_link = True
        wall_length = self.env_width / self.n_grid_segments

        self.asset_wall_x = self.gym.create_box(
            self.handle, wall_length-WALL_WIDTH, WALL_WIDTH, WALL_HEIGHT, static_asset_options)

        self.asset_wall_y = self.gym.create_box(
            self.handle, WALL_WIDTH, wall_length-WALL_WIDTH, WALL_HEIGHT, static_asset_options)

        self.asset_link = gym.create_box(
            self.handle, WALL_WIDTH, WALL_WIDTH, WALL_HEIGHT, static_asset_options)

        self.asset_object = gym.create_sphere(
            self.handle, OBJECT_RADIUS, static_asset_options)

        # Build envs from generated and loaded data
        self.envs = [MazeEnv(self, data_i, i) for i, data_i in enumerate(maze_data)]

        # Gather env data
        self.all_wallgrid_pairs = np.stack([env.data.grid_wall_pairs for env in self.envs])

        self.env_bot_handles = np.array([
            (env.handle, bot_handle)
            for env in self.envs
            for bot_handle in env.bot_handles], dtype=object)

        # Data for env resetting
        n_wallx = np.prod(self.constructor.hor_grid.shape[:-1])
        n_links = np.prod(self.constructor.link_grid.shape[:-1])

        self.wall_substate = np.concatenate((
            np.zeros((n_wallx, 3), dtype=np.float32),
            np.ones((n_wallx, 1), dtype=np.float32),
            np.zeros((n_wallx, 6), dtype=np.float32)), axis=-1, dtype=np.float32)

        self.link_substate = np.concatenate((
            np.zeros((n_links, 3), dtype=np.float32),
            np.ones((n_links, 1), dtype=np.float32),
            np.zeros((n_links, 6), dtype=np.float32)), axis=-1, dtype=np.float32)

        self.obj_substate = np.concatenate((
            np.ones((self.n_objects, 1), dtype=np.float32) * OBJECT_HEIGHT,
            np.zeros((self.n_objects, 3), dtype=np.float32),
            np.ones((self.n_objects, 1), dtype=np.float32),
            np.zeros((self.n_objects, 6), dtype=np.float32)), axis=-1, dtype=np.float32)

        self.bot_substates = (
            np.zeros((self.n_bots, 1), dtype=np.float32),
            np.zeros((self.n_bots, 6), dtype=np.float32))

    def reset(self, rst_env_indices: 'list[int]', full: bool = True) -> 'tuple[ndarray, ndarray]':
        actor_states = [self.envs[i].reset(full) for i in rst_env_indices]

        if full:
            for i in rst_env_indices:
                self.all_wallgrid_pairs[i] = self.envs[i].data.grid_wall_pairs

            env_idx_keys = ('wally_indices', 'wallx_indices', 'link_indices', 'obj_indices', 'bot_indices')

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


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    parser = ArgumentParser(description='Maze env. gen. inspection.')

    parser.add_argument('--level', type=int, default=4)
    parser.add_argument('--rng_seed', type=int, default=None)
    parser.add_argument('--bot_idx', type=int, default=0)
    parser.add_argument('--obj_idx', type=int, default=0)

    args = parser.parse_args()

    lvl: int = args.level
    seed: 'int | None' = args.rng_seed
    bot_idx: int = args.bot_idx
    obj_idx: int = args.obj_idx

    kwargs = LEVEL_PARAMS[lvl]
    kwargs['rng'] = kwargs['rng_seed'] if seed is None else seed
    del kwargs['rng_seed']
    del kwargs['ep_duration']

    cons = MazeConstructor(**kwargs)
    data = cons.generate()

    # Setup
    obj_clrs = COLOURS['basic'][data.obj_clr_idcs]
    wall_colours = COLOURS['pastel'][data.wall_clr_idcs]

    hor_wall_pos = cons.hor_grid.reshape(-1, 2)
    ver_wall_pos = cons.ver_grid.reshape(-1, 2)
    wall_halflen = abs(cons.grid_delims[0] - cons.grid_delims[1]) / 2.
    full_grid_edges = np.concatenate((
        np.stack((hor_wall_pos - (0., wall_halflen), hor_wall_pos + (0., wall_halflen)), axis=1),
        np.stack((ver_wall_pos - (wall_halflen, 0.), ver_wall_pos + (wall_halflen, 0.)), axis=1)))

    grid_edges = data.grid_wall_pairs.reshape(-1, 2, 2)
    grid_edges = grid_edges[np.any((grid_edges[:, 0, :] != grid_edges[:, 1, :]), axis=1)]
    link_points = np.unique(grid_edges.reshape(-1, 2), axis=0)

    conn_graph_points = data.grid_square_centres[[
        i for i in range(len(data.grid_square_centres))
        if len(data.grid_conn_map[i]) > 0]]

    conn_graph_edges = np.array([
        (data.grid_square_centres[start_idx], data.grid_square_centres[end_idx])
        for start_idx, neigh_idcs in data.grid_conn_map.items() for end_idx in neigh_idcs])

    start_pt = data.bot_spawn_points[bot_idx]
    end_pt = data.obj_points[obj_idx]

    path_pts, dist, _ = data.reconstruct_path(start_pt, end_pt)

    entry_pt = path_pts[-1]
    exit_pt = path_pts[0]

    # Plot
    _, axes = plt.subplots(2, 2, figsize=(10, 8))
    (ax0, ax1, ax2, ax3) = axes.flatten()

    # Wall clusters
    for edge, clr in zip(full_grid_edges, wall_colours):
        ax0.plot(edge[:, 0], edge[:, 1], color=clr, linewidth=3)

    ax0.set_facecolor('grey')

    # Base graph
    ax1.plot(grid_edges[..., 0].T, grid_edges[..., 1].T, 'b')
    ax1.plot(link_points[:, 0], link_points[:, 1], 'b.', mfc='none')

    ax1.plot(data.urq_graph_edges[..., 0].T, data.urq_graph_edges[..., 1].T, 'r')
    ax1.plot(data.urq_graph_points[:, 0], data.urq_graph_points[:, 1], 'k.')

    ax1.plot(data.bot_spawn_points[:, 0], data.bot_spawn_points[:, 1], 'ko')
    ax1.plot(data.bot_spawn_points[:, 0], data.bot_spawn_points[:, 1], 'w2')

    for pt, clr in zip(data.obj_points, obj_clrs):
        ax1.plot(pt[None, 0], pt[None, 1], 's', color=clr)

    # Connection graph
    ax2.plot(grid_edges[..., 0].T, grid_edges[..., 1].T, 'b')
    ax2.plot(conn_graph_edges[..., 0].T, conn_graph_edges[..., 1].T, 'r')
    ax2.plot(conn_graph_points[:, 0], conn_graph_points[:, 1], 'k.')

    for pt, clr in zip(data.obj_points, obj_clrs):
        ax2.plot(pt[None, 0], pt[None, 1], 's', color=clr)

    # A* path
    ax3.plot(grid_edges[..., 0].T, grid_edges[..., 1].T, 'b')

    ax3.plot(path_pts[:, 0], path_pts[:, 1], 'r')
    ax3.plot([start_pt[0], entry_pt[0]], [start_pt[1], entry_pt[1]], 'g')
    ax3.plot([exit_pt[0], end_pt[0]], [exit_pt[1], end_pt[1]], 'g')

    for pt, clr in zip(data.obj_points, obj_clrs):
        ax3.plot(pt[None, 0], pt[None, 1], 's', color=clr)

    ax3.plot(start_pt[None, 0], start_pt[None, 1], 'kx')
    ax3.plot(end_pt[None, 0], end_pt[None, 1], 'wx')

    ax0.set_title('Wall clusters')
    ax1.set_title('Urquhart graph')
    ax2.set_title('Connection graph')
    ax3.set_title(f'A* path: {dist:.2f}')

    plt.show()
