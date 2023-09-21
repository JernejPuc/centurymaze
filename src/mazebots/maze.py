"""Procedural generation of maze-like environments"""

from argparse import ArgumentParser

import numpy as np
from numpy import ndarray
from scipy.cluster.vq import kmeans2

import config as cfg
from utils import (
    any_intersections,
    astar,
    eval_path,
    get_cached_paths,
    get_numba_dict,
    min_distance,
    prune_path_backward,
    prune_path_forward,
    reconstruct_paths,
    urquhart)


class MazeData:
    """Describes the topology and functional elements of a specific environment."""

    con_graph_points: ndarray
    con_graph_edges: ndarray

    regcon_graph_points: ndarray
    regcon_graph_map: 'dict[int, ndarray[int]]'
    regcon_graph_paths: 'dict[tuple[int, int], ndarray[int]]'

    grid_delims: ndarray
    open_grid_delims: ndarray
    grid_square_centres: ndarray
    grid_wall_pairs: ndarray

    hor_wall_mask: ndarray
    ver_wall_mask: ndarray
    sqr_link_mask: ndarray
    sqr_roof_mask: ndarray

    obj_points: ndarray
    obj_trans_probs: ndarray
    obj_clr_idcs: ndarray

    hor_wall_clr_idcs: ndarray
    ver_wall_clr_idcs: ndarray
    sqr_roof_clr_idcs: ndarray

    bot_spawn_points: ndarray
    bot_spawn_angles: ndarray

    subenv_idcs: 'tuple[int, int]'
    subenv_idx_list: 'list[tuple[int, int]]'

    def __init__(self, constructor: 'MazeConstructor', precompute: bool = True, **kwargs):
        self.constructor = constructor
        self.grid_delims = constructor.grid_delims
        self.open_grid_delims = constructor.open_grid_delims

        # NOTE: Constructor views level params. as the points of reference
        self.n_grid_segments = constructor.n_supgrid_segments
        self.n_subgrid_segments = constructor.n_grid_segments

        for key, val in kwargs.items():
            setattr(self, key, val)

        if 'subenv_idcs' not in kwargs:
            self.subenv_idcs = (0, 0)

        if 'subenv_idx_list' not in kwargs:
            self.init_subenv_indices()

        if 'regcon_graph_map' not in kwargs or 'regcon_graph_points' not in kwargs:
            self.init_connection_map()

        if 'regcon_graph_paths' not in kwargs and 'obj_points' in kwargs:
            self.init_path_map(precompute=precompute)

    def copy(self) -> 'MazeData':
        kwargs = {}

        for k, v in self.__dict__.items():
            if k == 'constructor':
                continue

            if isinstance(v, ndarray):
                kwargs[k] = v.copy()

            else:
                kwargs[k] = v

        return MazeData(self.constructor, **kwargs)

    def init_subenv_indices(self):
        """
        Get the upper-left indices of all subenvs. within a larger env. area
        which have enough open squares and no disconnected paths between them.
        """

        if self.n_subgrid_segments == self.n_grid_segments:
            self.subenv_idx_list = [self.subenv_idcs]
            return

        def spread_open_squares(sqr_open_mask: ndarray, hor_wall_mask: ndarray, ver_wall_mask: ndarray, i: int, j: int):
            """
            Recursively mark open squares within the given env. area.
            If all open squares are connected (reachable from within this area),
            they will all be marked.
            """

            # Mark mask at current indices
            sqr_open_mask[i, j] = 1

            # Try spreading right
            if (j+1) < sqr_open_mask.shape[1] and sqr_open_mask[i, j+1] != 1 and not ver_wall_mask[i, j+1]:
                spread_open_squares(sqr_open_mask, hor_wall_mask, ver_wall_mask, i, j+1)

            # Try spreading down
            if (i+1) < sqr_open_mask.shape[0] and sqr_open_mask[i+1, j] != 1 and not hor_wall_mask[i+1, j]:
                spread_open_squares(sqr_open_mask, hor_wall_mask, ver_wall_mask, i+1, j)

            # Try spreading left
            if (j-1) >= 0 and sqr_open_mask[i, j-1] != 1 and not ver_wall_mask[i, j]:
                spread_open_squares(sqr_open_mask, hor_wall_mask, ver_wall_mask, i, j-1)

            # Try spreading up
            if (i-1) >= 0 and sqr_open_mask[i-1, j] != 1 and not hor_wall_mask[i, j]:
                spread_open_squares(sqr_open_mask, hor_wall_mask, ver_wall_mask, i-1, j)

        self.subenv_idx_list = []

        sqr_open_mask = ~self.sqr_roof_mask
        n_segments = self.n_subgrid_segments
        n_min_open_squares = round(n_segments**2 * 0.6975)

        for i in range(sqr_open_mask.shape[0] - (n_segments-1)):
            for j in range(sqr_open_mask.shape[1] - (n_segments-1)):
                sqr_open_submask = sqr_open_mask[i:i+n_segments, j:j+n_segments]
                n_open_squares = sqr_open_submask.sum()

                if n_open_squares < n_min_open_squares:
                    continue

                sqr_con_open_submask = np.zeros((n_segments, n_segments), dtype=np.int64)

                hor_wall_submask = self.hor_wall_mask[i:i+n_segments+1, j:j+n_segments]
                ver_wall_submask = self.ver_wall_mask[i:i+n_segments, j:j+n_segments+1]

                sqr_open_idcs = np.nonzero(sqr_open_submask)
                start_i, start_j = sqr_open_idcs[0][0], sqr_open_idcs[1][0]

                spread_open_squares(sqr_con_open_submask, hor_wall_submask, ver_wall_submask, start_i, start_j)
                n_con_open_squares = sqr_con_open_submask.sum()

                if n_open_squares == n_con_open_squares:
                    self.subenv_idx_list.append((i, j))

    def init_connection_map(self):
        """Build a graph of connections between connected squares on the grid."""

        n_segments = self.n_grid_segments

        hor_con_mask = ~self.hor_wall_mask
        ver_con_mask = ~self.ver_wall_mask

        delim_centres = self.constructor.grid_delim_centres

        regcon_graph_map: 'dict[int, list[int]]' = {
            (i*n_segments+j): []
            for i in range(n_segments)
            for j in range(n_segments)}

        regcon_graph_edges: 'list[tuple[tuple[float, float], tuple[float, float]]]' = []

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
                    regcon_graph_map[og_node].append(hor_node)
                    regcon_graph_map[hor_node].append(og_node)

                    edge = ((delim_centres[i], delim_centres[j]), (delim_centres[i], delim_centres[j_post]))
                    regcon_graph_edges.append(edge)

                if ver_con:
                    regcon_graph_map[og_node].append(ver_node)
                    regcon_graph_map[ver_node].append(og_node)

                    edge = ((delim_centres[i], delim_centres[j]), (delim_centres[i_post], delim_centres[j]))
                    regcon_graph_edges.append(edge)

                if ldiag_con:
                    regcon_graph_map[og_node].append(ldiag_node)
                    regcon_graph_map[ldiag_node].append(og_node)

                    edge = ((delim_centres[i], delim_centres[j]), (delim_centres[i_post], delim_centres[j_post]))
                    regcon_graph_edges.append(edge)

                if udiag_con:
                    regcon_graph_map[og_node].append(udiag_node)
                    regcon_graph_map[udiag_node].append(og_node)

                    edge = ((delim_centres[i], delim_centres[j]), (delim_centres[i_prev], delim_centres[j_post]))
                    regcon_graph_edges.append(edge)

        # Convert connection graph into numba typed dict
        self.regcon_graph_map = get_numba_dict()

        for key, lst in regcon_graph_map.items():
            self.regcon_graph_map[key] = np.array(lst, dtype=np.int64)

        self.regcon_graph_points = np.unique(np.array(regcon_graph_edges).reshape(-1, 2), axis=0)

    def init_path_map(self, reset: bool = True, precompute: bool = True):
        """
        Use A* on the underlying graph to precompute estimated paths
        from all valid entry points to all target objects.
        """

        if reset:
            self.regcon_graph_paths = get_numba_dict(tuple_as_key=True)

        if not precompute:
            return

        exit_idcs = np.digitize(self.obj_points, self.open_grid_delims)
        flattened_exit_idcs = self.n_grid_segments * exit_idcs[:, 0] + exit_idcs[:, 1]

        for entry_node, v in self.regcon_graph_map.items():
            if len(v) == 0:
                continue

            for i, exit_node in enumerate(flattened_exit_idcs):
                path = astar(self.regcon_graph_map, self.grid_square_centres, entry_node, exit_node)

                self.regcon_graph_paths[entry_node, exit_node] = prune_path_backward(
                    path,
                    exit_idcs[i],
                    self.obj_points[i],
                    self.grid_square_centres,
                    self.grid_wall_pairs,
                    self.n_grid_segments)

    def get_path_estimate(
        self,
        start_pts: ndarray,
        end_pts: ndarray,
        sight_mask: ndarray,
        use_cache: bool = False
    ) -> 'tuple[ndarray, ndarray]':
        """
        Estimate path length and starting direction from valid starting points
        to target end points based on precomputed A* reference paths.
        """

        if use_cache:
            return get_cached_paths(
                start_pts,
                end_pts,
                sight_mask,
                self.regcon_graph_map,
                self.regcon_graph_paths,
                self.open_grid_delims,
                self.grid_square_centres,
                self.grid_wall_pairs)

        return reconstruct_paths(
            start_pts,
            end_pts,
            sight_mask,
            self.regcon_graph_paths,
            self.open_grid_delims,
            self.grid_square_centres,
            self.grid_wall_pairs)

    def reconstruct_path(self, start_pt: ndarray, end_pt: ndarray) -> 'tuple[ndarray, float, ndarray]':
        """Estimate path length and starting direction for a single test case."""

        entry_idcs = np.digitize(start_pt, self.open_grid_delims)
        exit_idcs = np.digitize(end_pt, self.open_grid_delims)

        entry_node = self.n_grid_segments * entry_idcs[0] + entry_idcs[1]
        exit_node = self.n_grid_segments * exit_idcs[0] + exit_idcs[1]

        path = prune_path_forward(
            self.regcon_graph_paths[entry_node, exit_node],
            entry_idcs,
            start_pt,
            self.grid_square_centres,
            self.grid_wall_pairs,
            self.n_grid_segments)

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
        rng: 'None | int | np.random.Generator' = None,
        supenv_width: int = None,
        n_supgrid_segments: int = None
    ):
        self.env_width = env_width
        self.env_halfwidth = env_width / 2.
        self.n_grid_segments = n_grid_segments

        if supenv_width is None:
            supenv_width = env_width if n_supgrid_segments is None else (n_supgrid_segments * cfg.WALL_LENGTH)

        if n_supgrid_segments is None:
            n_supgrid_segments = n_grid_segments if supenv_width is env_width else round(supenv_width / cfg.WALL_LENGTH)

        self.supenv_halfwidth = supenv_width / 2.
        self.n_supgrid_segments = n_supgrid_segments

        self.min_object_spacing = max(env_width / 3., cfg.GOAL_RADIUS*2 + cfg.MIN_BUFFER)

        # Grid delimiters
        self.grid_delims = np.linspace(-self.supenv_halfwidth, self.supenv_halfwidth, n_supgrid_segments+1)
        self.open_grid_delims = self.grid_delims[1:-1]
        self.grid_delim_centres = np.round((self.grid_delims[1:] + self.grid_delims[:-1]) / 2., 6)

        y_roof, x_roof = np.meshgrid(self.grid_delim_centres, self.grid_delim_centres)
        y_link, x_link = np.meshgrid(self.grid_delims, self.grid_delims)
        y_hor, x_hor = np.meshgrid(self.grid_delim_centres, self.grid_delims)
        y_ver, x_ver = np.meshgrid(self.grid_delims, self.grid_delim_centres)

        # Link and wall centres
        self.roof_grid = np.stack((x_roof, y_roof), axis=-1)
        self.link_grid = np.stack((x_link, y_link), axis=-1)
        self.hor_grid = np.stack((x_hor, y_hor), axis=-1)
        self.ver_grid = np.stack((x_ver, y_ver), axis=-1)

        self.n_graph_points = n_graph_points
        self.n_bots = n_bots
        self.n_objects = n_objects

        if isinstance(rng, np.random.Generator):
            self.rng = rng
            self.same_state_init = False

        else:
            self.rng = np.random.default_rng(rng)
            self.same_state_init = True

        self.rng_state_init = self.rng.__getstate__().copy()

    def reset_rng(self) -> 'ndarray | None':
        if self.same_state_init:
            rng_state = self.rng.__getstate__().copy()
            self.rng.__setstate__(self.rng_state_init)

        else:
            rng_state = None

        return rng_state

    def resume_rng(self, rng_state: 'ndarray | None'):
        if self.same_state_init:
            self.rng.__setstate__(rng_state)

    def prune_grid(
        self,
        con_graph_edges: ndarray
    ) -> 'tuple[ndarray, ndarray, ndarray, ndarray, ndarray]':
        """Mask edges that intersect any edge of the given graph."""

        n_delims = len(self.grid_delims)
        n_segments = self.n_supgrid_segments

        # NOTE: [(0., 0.), (0., 0.)] is considered a null edge (clearing)
        grid_wall_pairs = np.zeros((n_delims, n_delims, 2, 2, 2))

        sqr_link_mask = np.zeros((n_delims, n_delims), dtype=np.bool8)
        hor_wall_mask = np.zeros((n_delims, n_segments), dtype=np.bool8)
        ver_wall_mask = np.zeros((n_segments, n_delims), dtype=np.bool8)

        wall_edges: 'list[ndarray]' = []

        # Iterate over pairs of edges in the grid
        for i in range(n_delims):
            for j in range(n_delims):

                # Check for intersections
                if j < n_segments:
                    hor_edge = self.link_grid[i, j:j+2]

                    if not any_intersections(*hor_edge, con_graph_edges):
                        sqr_link_mask[i, j:j+2] = True
                        hor_wall_mask[i, j] = True
                        grid_wall_pairs[i, j, 0] = hor_edge
                        wall_edges.append(hor_edge)

                if i < n_segments:
                    ver_edge = self.link_grid[i:i+2, j]

                    if not any_intersections(*ver_edge, con_graph_edges):
                        sqr_link_mask[i:i+2, j] = True
                        ver_wall_mask[i, j] = True
                        grid_wall_pairs[i, j, 1] = ver_edge
                        wall_edges.append(ver_edge)

        wall_centres = np.array(wall_edges).mean(axis=1)

        # Roofs are placed wherever all four underlying walls are raised
        sqr_roof_mask = np.zeros((n_segments, n_segments), dtype=np.bool_)

        for i in range(n_segments):
            for j in range(n_segments):
                sqr_roof_mask[i, j] = (hor_wall_mask[i:i+2, j].sum() + ver_wall_mask[i, j:j+2].sum()) == 4

        return wall_centres, grid_wall_pairs, sqr_roof_mask, sqr_link_mask, hor_wall_mask, ver_wall_mask

    def generate(self, data: 'None | MazeData' = None) -> MazeData:
        """Randomly generate mazes until the underlying graph is feasible."""

        if data is not None:
            return self.reposition(data)

        while True:
            rng_state = self.reset_rng()

            # Generate graph vertices
            con_graph_points = self.rng.uniform(
                low=-self.env_halfwidth,
                high=self.env_halfwidth,
                size=(self.n_graph_points, 2))

            self.resume_rng(rng_state)

            # Construct a cyclical, connected graph
            con_graph_edges = urquhart(con_graph_points)

            # Create corridors where the graph's edges overlap with the grid
            (
                wall_centres,
                grid_wall_pairs,
                sqr_roof_mask,
                sqr_link_mask,
                hor_wall_mask,
                ver_wall_mask) = self.prune_grid(con_graph_edges)

            # Spawn points must lie on the graph (vertices and points along the edges)
            candidate_points = np.concatenate((con_graph_points, np.mean(con_graph_edges, axis=1)))

            # Try to designate n_objects + n_bots points on the maze graph
            # as objectives and bot spawn points, respectively
            # Reconstruct the graph until all objects and spawn points are selected
            try:
                object_points = self.select_points(
                    candidate_points,
                    self.n_objects, self.min_object_spacing,
                    None, 0.,
                    grid_wall_pairs, cfg.OBJECT_TO_WALL_BUFFER,
                    self.MAX_OBJECT_SHUFFLES)

                bot_spawn_points = self.select_points(
                    candidate_points,
                    self.n_bots, cfg.BOT_TO_BOT_BUFFER,
                    object_points, cfg.BOT_TO_OBJECT_BUFFER,
                    grid_wall_pairs, cfg.BOT_TO_WALL_BUFFER,
                    self.MAX_BOT_SHUFFLES)

                break

            except AssertionError:
                pass

        # Sample unique object colours
        object_colour_idx_order = self.rng.permutation(cfg.N_OBJ_COLOURS)
        object_colour_idcs = object_colour_idx_order[:self.n_objects]

        # Sample bot spawn angles uniformly
        bot_spawn_angles = self.rng.uniform(low=-np.pi, high=np.pi, size=self.n_bots)

        # Sample objective transition probabilities
        object_trans_probs = 0.8 + 0.2 * self.rng.uniform(size=(self.n_objects, self.n_objects))
        object_trans_probs[np.diag_indices_from(object_trans_probs)] = 0.
        object_trans_probs /= object_trans_probs.sum(axis=-1)[..., None]

        rng_state = self.reset_rng()

        # Sample wall colours
        wall_colour_idx_order = self.rng.permutation(cfg.N_WALL_COLOURS)

        # Assign wall colours in clusters based on position
        hor_shape, ver_shape = self.hor_grid.shape[:2], self.ver_grid.shape[:2]
        wall_positions = np.vstack((self.hor_grid.reshape(-1, 2), self.ver_grid.reshape(-1, 2)))

        # NOTE: Empty clusters (not associated with any point) are auto-resolved with random points
        centroids = (
            wall_centres
            if len(wall_centres) <= cfg.N_WALL_COLOURS
            else kmeans2(wall_centres, cfg.N_WALL_COLOURS, seed=self.rng)[0])

        closest_centroid_idx = np.argmin(np.linalg.norm(wall_positions[None] - centroids[:, None], axis=-1), axis=0)
        wall_colour_idcs = wall_colour_idx_order[closest_centroid_idx]

        hor_wall_clr_idcs, ver_wall_clr_idcs = np.split(wall_colour_idcs, (hor_shape[0] * hor_shape[1],))
        hor_wall_clr_idcs = hor_wall_clr_idcs.reshape(hor_shape)
        ver_wall_clr_idcs = ver_wall_clr_idcs.reshape(ver_shape)

        n_squares_per_row = self.n_supgrid_segments
        sqr_roof_clr_idcs = np.zeros((n_squares_per_row, n_squares_per_row), dtype=np.int64)

        # Roofs inherit the majority class of the underlying wall colours
        for i in range(n_squares_per_row):
            for j in range(n_squares_per_row):
                sqr_roof_clr_idcs[i, j] = np.argmax(np.bincount(
                    np.concatenate((hor_wall_clr_idcs[i:i+2, j], ver_wall_clr_idcs[i, j:j+2])),
                    minlength=cfg.N_WALL_COLOURS))

        self.resume_rng(rng_state)

        # Node positions for A*
        grid_square_centres = self.roof_grid.reshape(-1, 2)

        return MazeData(
            self,
            con_graph_points=con_graph_points,
            con_graph_edges=con_graph_edges,
            grid_square_centres=grid_square_centres,
            grid_wall_pairs=grid_wall_pairs,
            hor_wall_mask=hor_wall_mask,
            ver_wall_mask=ver_wall_mask,
            sqr_link_mask=sqr_link_mask,
            sqr_roof_mask=sqr_roof_mask,
            obj_points=object_points,
            obj_trans_probs=object_trans_probs,
            obj_clr_idcs=object_colour_idcs,
            hor_wall_clr_idcs=hor_wall_clr_idcs,
            ver_wall_clr_idcs=ver_wall_clr_idcs,
            sqr_roof_clr_idcs=sqr_roof_clr_idcs,
            bot_spawn_points=bot_spawn_points,
            bot_spawn_angles=bot_spawn_angles)

    def reposition(self, data: MazeData) -> MazeData:
        """Move spawns and objects around, but keep the overall maze structure."""

        while True:
            candidate_points, data.subenv_idcs = self.sample_env_points(
                data,
                n_to_sample=(self.n_objects + self.n_bots) * 3,
                max_dist_to_graph_pts=cfg.WALL_HALFLENGTH - cfg.OBJECT_TO_WALL_BUFFER)

            try:
                data.obj_points = self.select_points(
                    candidate_points,
                    self.n_objects, self.min_object_spacing,
                    None, 0.,
                    data.grid_wall_pairs, cfg.OBJECT_TO_WALL_BUFFER,
                    self.MAX_OBJECT_SHUFFLES)

                data.bot_spawn_points = self.select_points(
                    candidate_points,
                    self.n_bots, cfg.BOT_TO_BOT_BUFFER,
                    data.obj_points, cfg.BOT_TO_OBJECT_BUFFER,
                    data.grid_wall_pairs, cfg.BOT_TO_WALL_BUFFER,
                    self.MAX_BOT_SHUFFLES)

                break

            except AssertionError:
                pass

        obj_trans_probs = 0.8 + 0.2 * self.rng.uniform(size=(self.n_objects, self.n_objects))
        obj_trans_probs[np.diag_indices_from(obj_trans_probs)] = 0.
        obj_trans_probs /= obj_trans_probs.sum(axis=-1)[..., None]

        data.obj_trans_probs = obj_trans_probs
        data.obj_clr_idcs = self.rng.permutation(cfg.N_OBJ_COLOURS)[:self.n_objects]
        data.bot_spawn_angles = self.rng.uniform(low=-np.pi, high=np.pi, size=self.n_bots)

        data.init_path_map(reset='regcon_graph_paths' not in data.__dict__, precompute=False)

        return data

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

            self.rng.shuffle(candidate_points)

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
                    sqr_idx = max(1, min(len(self.open_grid_delims), sqr_idx))
                    sqr_idy = max(1, min(len(self.open_grid_delims), sqr_idy))

                    edge_subset = grid_edges[sqr_idx-1:sqr_idx+2, sqr_idy-1:sqr_idy+2].reshape(-1, 2, 2)

                    if any(min_distance(*edge, point) < min_dist_to_grid_edges for edge in edge_subset):
                        continue

                # Assign selected point
                points[last_idx] = point
                last_idx += 1

                if last_idx == n_to_select:
                    return points

        raise AssertionError

    def sample_env_points(
        self,
        data: MazeData,
        n_to_sample: int,
        max_dist_to_graph_pts: float
    ) -> 'tuple[ndarray, tuple[int, int]]':
        """
        Prune nodes of the regular connection graph if they exceed subenv.
        boundaries, then sample points around the remaining nodes.
        """

        pts = None

        # Nothing to prune
        if len(data.subenv_idx_list) == 1:
            i, j = env_idcs = data.subenv_idx_list[0]

            if i == j == 0:
                pts = data.regcon_graph_points

        # Sample env. and filter nodes
        if pts is None:
            i, j = env_idcs = self.rng.choice(data.subenv_idx_list)
            pts = data.regcon_graph_points

            min_x = -self.supenv_halfwidth + i * cfg.WALL_LENGTH
            max_x = min_x + self.env_width

            min_y = -self.supenv_halfwidth + j * cfg.WALL_LENGTH
            max_y = min_y + self.env_width

            pts = pts[(pts[:, 0] > min_x) & (pts[:, 0] < max_x) & (pts[:, 1] > min_y) & (pts[:, 1] < max_y)]

        # Sample nodes from within the env., with replacement
        candidate_indices = self.rng.choice(len(pts), n_to_sample, replace=True)

        # Get more varied points by adding random offsets to sampled nodes
        candidate_points = pts[candidate_indices] + \
            self.rng.uniform(-max_dist_to_graph_pts, max_dist_to_graph_pts, (n_to_sample, 2))

        return candidate_points, env_idcs


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    parser = ArgumentParser(description='Maze env. gen. inspection.')

    parser.add_argument('--level', type=int, default=4)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--rng_seed', type=int, default=None)
    parser.add_argument('--bot_idx', type=int, default=0)
    parser.add_argument('--obj_idx', type=int, default=0)

    args = parser.parse_args()

    lvl: int = args.level
    data_path: 'str | None' = args.data_path
    seed: 'int | None' = args.rng_seed
    bot_idx: int = args.bot_idx
    obj_idx: int = args.obj_idx

    kwargs = cfg.LEVEL_PARAMS[lvl]
    kwargs['rng'] = kwargs['rng_seed'] if seed is None else seed
    del kwargs['rng_seed']
    del kwargs['ep_duration']

    if data_path is None:
        cons = MazeConstructor(**kwargs)
        data = cons.generate()

    else:
        data_dict = np.load(data_path)
        supgrid_delims = data_dict['grid_delims']
        supenv_width = supgrid_delims[-1] - supgrid_delims[0]
        n_supgrid_segments = len(supgrid_delims) - 1

        cons = MazeConstructor(**kwargs, supenv_width=supenv_width, n_supgrid_segments=n_supgrid_segments)
        data = cons.generate(MazeData(cons, **data_dict))
        data.init_path_map(reset=False)

    # Setup
    subenv_slice = (slice(data.subenv_idcs[0], None), slice(data.subenv_idcs[1], None))

    obj_clrs = np.array(cfg.COLOURS['basic'])[data.obj_clr_idcs]
    hor_wall_clrs = np.array(cfg.COLOURS['pastel'])[data.hor_wall_clr_idcs][subenv_slice]
    ver_wall_clrs = np.array(cfg.COLOURS['pastel'])[data.ver_wall_clr_idcs][subenv_slice]
    sqr_roof_clrs = np.array(cfg.COLOURS['pastel'])[data.sqr_roof_clr_idcs][subenv_slice]

    subenv_offsets = np.array(data.subenv_idcs) * cfg.WALL_LENGTH - cons.supenv_halfwidth

    hor_grid = cons.hor_grid[subenv_slice]
    ver_grid = cons.ver_grid[subenv_slice]
    roof_grid = cons.roof_grid[subenv_slice]

    wall_halflen = (((0., cfg.WALL_HALFLENGTH),),)
    hor_wall_edges = np.stack((hor_grid - wall_halflen, hor_grid + wall_halflen), axis=-1)
    wall_halflen = (((cfg.WALL_HALFLENGTH, 0.),),)
    ver_wall_edges = np.stack((ver_grid - wall_halflen, ver_grid + wall_halflen), axis=-1)

    grid_edges = data.grid_wall_pairs.reshape(-1, 2, 2)
    grid_edges = grid_edges[np.any((grid_edges[:, 0, :] != grid_edges[:, 1, :]), axis=1)]
    link_points = np.unique(grid_edges.reshape(-1, 2), axis=0)

    regcon_graph_edges = np.array([
        (data.grid_square_centres[start_idx], data.grid_square_centres[end_idx])
        for start_idx, neigh_idcs in data.regcon_graph_map.items() for end_idx in neigh_idcs])

    start_pt = data.bot_spawn_points[bot_idx]
    end_pt = data.obj_points[obj_idx]

    path_pts, dist, _ = data.reconstruct_path(start_pt, end_pt)

    entry_pt = path_pts[-1]
    exit_pt = path_pts[0]

    # Plot
    _, axes = plt.subplots(2, 2, figsize=(13, 13))
    (ax0, ax1, ax2, ax3) = axes.flatten()

    # Wall clusters
    n_rows, n_cols = hor_grid.shape[:-1]
    mask = data.hor_wall_mask[subenv_slice]

    for i in range(n_rows):
        for j in range(n_cols):
            if mask[i, j]:
                edge_x, edge_y = hor_wall_edges[i, j]
                ax0.plot(edge_x, edge_y, color=hor_wall_clrs[i, j], linewidth=3)

    n_rows, n_cols = ver_grid.shape[:-1]
    mask = data.ver_wall_mask[subenv_slice]

    for i in range(n_rows):
        for j in range(n_cols):
            if mask[i, j]:
                edge_x, edge_y = ver_wall_edges[i, j]
                ax0.plot(edge_x, edge_y, color=ver_wall_clrs[i, j], linewidth=3)

    n_rows, n_cols = roof_grid.shape[:-1]
    mask = data.sqr_roof_mask[subenv_slice]

    for i in range(n_rows):
        for j in range(n_cols):
            if mask[i, j]:
                edge_x, edge_yu = ver_wall_edges[i, j]
                _, edge_yl = ver_wall_edges[i, j+1]
                ax0.fill_between(edge_x, edge_yl, edge_yu, color=sqr_roof_clrs[i, j])

    ax0.set_facecolor('grey')
    ax0.set_xlim((subenv_offsets[0] - 5*cfg.WALL_WIDTH, subenv_offsets[0] + 5*cfg.WALL_WIDTH + cons.env_width))
    ax0.set_ylim((subenv_offsets[1] - 5*cfg.WALL_WIDTH, subenv_offsets[1] + 5*cfg.WALL_WIDTH + cons.env_width))

    # Base graph
    ax1.plot(grid_edges[..., 0].T, grid_edges[..., 1].T, 'b')
    ax1.plot(link_points[:, 0], link_points[:, 1], 'b.', mfc='none')

    ax1.plot(data.con_graph_edges[..., 0].T, data.con_graph_edges[..., 1].T, 'r')
    ax1.plot(data.con_graph_points[:, 0], data.con_graph_points[:, 1], 'k.')

    ax1.plot(data.bot_spawn_points[:, 0], data.bot_spawn_points[:, 1], 'ko')
    ax1.plot(data.bot_spawn_points[:, 0], data.bot_spawn_points[:, 1], 'w2')

    for pt, clr in zip(data.obj_points, obj_clrs):
        ax1.plot(pt[None, 0], pt[None, 1], 's', color=clr)

    # Connection graph
    ax2.plot(grid_edges[..., 0].T, grid_edges[..., 1].T, 'b')
    ax2.plot(regcon_graph_edges[..., 0].T, regcon_graph_edges[..., 1].T, 'r')
    ax2.plot(data.regcon_graph_points[:, 0], data.regcon_graph_points[:, 1], 'k.')

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

    ax0.set_title('Subenv. clusters')
    ax1.set_title('Base connection graph')
    ax2.set_title('Reg. connection graph')
    ax3.set_title(f'Reg. A* path ex.: {dist:.2f}')

    plt.show()
