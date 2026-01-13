"""Maze initialisation"""

from math import ceil

import numpy as np
from numpy import ndarray
from scipy.cluster.vq import kmeans2

import config as cfg
from utils import (
    any_intersections,
    get_cached_paths,
    get_numba_dict,
    min_distance,
    prune_path_forward,
    ray_trace,
    urquhart)


# ------------------------------------------------------------------------------
# MARK: MazeData

class MazeData:
    """Describes the topology and functional elements of a specific environment."""

    spawn_pts: ndarray
    spawn_angles: ndarray
    speaker_mask: ndarray
    decoy_pts: ndarray
    obj_pts: ndarray
    bot_obj_map: ndarray
    obj_goal_map: ndarray
    bot_goal_map: ndarray
    target_cell_pts: ndarray
    free_cell_pts: ndarray
    free_cell_mask: ndarray
    target_cell_mask: ndarray
    spawn_cell_mask: ndarray
    wall_mask: ndarray
    block_mask: ndarray
    link_mask: ndarray
    link_clr_idcs: ndarray
    wall_clr_idcs: ndarray
    wall_clr_idx_map: ndarray
    cell_clr_idx_grid: ndarray
    cell_wall_grid: ndarray
    cell_pass_map: 'dict[int, list[int]]'
    global_spawn_flag: int
    ep_duration: int

    def __init__(self, **kwargs):
        for name, val in kwargs.items():
            setattr(self, name, val)


# --------------------------------------------------------------------------
# MARK: MazeConstructor

class MazeConstructor:
    """Keeps and processes data needed to set up random environments."""

    base_spawn_pts: ndarray
    sub_spawn_pts: ndarray
    spawn_cell_mask: ndarray
    exterior_cell_mask: ndarray

    def __init__(
        self,
        side_length: int,
        wall_length: float,
        n_graph_points: int,
        n_bots: int,
        n_goals: int,
        ep_duration: float,
        n_decoys: int = 0,
        global_spawn_prob: float = 0.,
        speaker_dropout: float = 0.,
        rng: 'None | int | np.random.Generator' = None,
        level: int = None
    ):
        self.level = level

        self.side_length = side_length
        self.side_halflength = side_length / 2.

        self.wall_length = wall_length
        self.n_side_divs = int(side_length / wall_length)
        self.n_cell_divs = max(1, int(wall_length / (cfg.BOT_WIDTH * 1.5)))
        self.wall_halflength = wall_length / 2.
        self.min_goal_spacing = side_length / 3.

        # Objects are biased to spawn slightly to the side instead of directly at the middle of a corridor
        self.max_obj_to_wall_dist = min(wall_length / 3., wall_length - cfg.WALL_WIDTH - cfg.BOT_WIDTH - cfg.OBJ_HEIGHT)

        self.n_graph_points = n_graph_points
        self.n_bots = n_bots
        self.n_goals = n_goals
        self.n_decoys = n_decoys if n_decoys > 0 else n_goals
        self.n_bots_per_goal = self.n_bots / self.n_goals

        self.ep_duration = ep_duration

        self.global_spawn_prob = global_spawn_prob
        self.speaker_dropout = speaker_dropout
        self.rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

        # Grid delimiters
        self.delims = np.linspace(-self.side_halflength, self.side_halflength, self.n_side_divs+1)
        self.open_delims = self.delims[1:-1]
        midpts = np.round((self.delims[1:] + self.delims[:-1]) / 2, 2)

        self.delim_pt_grid = np.stack(np.meshgrid(self.delims, self.delims, indexing='ij'), axis=-1)
        self.cell_pt_grid = np.stack(np.meshgrid(midpts, midpts, indexing='ij'), axis=-1)

        self.wall_w_grid = np.stack(np.meshgrid(midpts, self.delims, indexing='ij'), axis=-1)
        self.wall_n_grid = np.stack(np.meshgrid(self.delims, midpts, indexing='ij'), axis=-1)

        self.init_candidate_points()

    # --------------------------------------------------------------------------
    # MARK: init_candidate_points

    def init_candidate_points(self):
        """Extract points of cells to be used for spawning or as targets."""

        cell_dist_grid = np.linalg.norm(self.cell_pt_grid, axis=-1)
        dist_sorted_cell_idcs = np.argsort(cell_dist_grid.ravel())

        n_spawn_cells = ceil(self.n_bots / self.n_cell_divs ** 2)
        n_buffer_cells = min(
            len(dist_sorted_cell_idcs) - (self.n_goals + self.n_decoys),
            min(self.n_side_divs**2 - 2, ceil((n_spawn_cells ** 0.5 + 2) ** 2)))

        spawn_idcs = dist_sorted_cell_idcs[:n_spawn_cells]
        buffer_idcs = dist_sorted_cell_idcs[:n_buffer_cells]

        self.spawn_cell_mask = np.zeros((self.n_side_divs,)*2, dtype=np.bool_)
        self.spawn_cell_mask.ravel()[spawn_idcs] = True
        self.base_spawn_pts = self.cell_pt_grid.reshape(-1, 2)[spawn_idcs]

        self.exterior_cell_mask = np.ones((self.n_side_divs,)*2, dtype=np.bool_)
        self.exterior_cell_mask.ravel()[buffer_idcs] = False

        # Sub-cell spawn points
        cell_spacing = self.side_length / self.n_side_divs
        cell_sub_spacing = self.side_length / (self.n_cell_divs * self.n_side_divs)

        offsets = np.linspace(-1., 1., self.n_cell_divs) * (cell_spacing - cell_sub_spacing) / 2.
        offsets = np.stack(np.meshgrid(offsets, offsets, indexing='ij'), axis=-1).reshape(-1, 1, 2)

        self.sub_spawn_pts = (self.base_spawn_pts.reshape(1, -1, 2) + offsets).reshape(-1, 2)

        # Sort from centre outwards
        spawn_pt_centroid = self.base_spawn_pts.mean(axis=0)
        spawn_pt_dists = np.linalg.norm(self.sub_spawn_pts - spawn_pt_centroid, axis=-1)
        self.sub_spawn_pts = self.sub_spawn_pts[np.argsort(spawn_pt_dists)]

    # --------------------------------------------------------------------------
    # MARK: refresh

    def refresh(self, data: MazeData):
        data.obj_pts = self.sample_viable_points(
            data.target_cell_pts, data.cell_wall_grid,
            self.n_goals, self.min_goal_spacing,
            None, 0.,
            cfg.OBJ_TO_WALL_GAP)

    # --------------------------------------------------------------------------
    # MARK: generate

    def generate(self) -> MazeData:
        """Randomly generate mazes until the underlying graph is feasible."""

        # Option to spawn out of the designated area for random data collection
        global_spawn_flag = self.global_spawn_prob and self.rng.random() < self.global_spawn_prob

        # Reconstruct the graph until all object points are selected
        while True:

            # Generate graph vertices
            graph_points = self.rng.uniform(
                low=-self.side_halflength,
                high=self.side_halflength,
                size=(self.n_graph_points, 2))

            # Ensure space in the centre of the maze
            graph_points = np.concatenate((graph_points, self.base_spawn_pts))

            # Construct a cyclical, connected graph
            graph_edges = urquhart(graph_points)

            # Create corridors where the graph's edges overlap with the grid
            link_mask, block_mask, wall_mask, cell_wall_grid = self.prune_grid(graph_edges)

            free_cell_mask = ~block_mask
            target_cell_mask = free_cell_mask & self.exterior_cell_mask
            free_cell_pts = self.cell_pt_grid[free_cell_mask]
            target_cell_pts = self.cell_pt_grid[target_cell_mask]

            if target_cell_pts.size == 0:
                continue

            # Try to designate n_goals points in free cells
            obj_pts = self.sample_viable_points(
                target_cell_pts, cell_wall_grid,
                self.n_goals, self.min_goal_spacing,
                None, 0.,
                cfg.OBJ_TO_WALL_GAP, self.max_obj_to_wall_dist)

            if obj_pts is not None:
                break

        # Clutter
        decoy_pts = self.sample_viable_points(
            target_cell_pts, cell_wall_grid,
            self.n_decoys, cfg.OBJ_TO_OBJ_GAP,
            obj_pts, cfg.OBJ_TO_OBJ_GAP,
            cfg.OBJ_TO_WALL_GAP, self.max_obj_to_wall_dist)

        # Pick points from spawn area or use pre-sorted points closest to the centroid of that area
        if global_spawn_flag:
            spawn_pts = self.sample_viable_points(
                free_cell_pts, cell_wall_grid,
                self.n_bots, cfg.BOT_TO_BOT_GAP,
                obj_pts, cfg.BOT_TO_OBJ_GAP,
                cfg.BOT_TO_WALL_GAP)

        else:
            spawn_pts = self.sub_spawn_pts[:self.n_bots]

        # Sample bot spawn angles uniformly
        spawn_angles = self.rng.uniform(low=-np.pi, high=np.pi, size=self.n_bots)

        # Sample wall colours (clustered blocks + the outer border)
        wall_clr_idx_map = self.rng.permutation(cfg.N_WALL_CLRS)

        # Assign wall colours in clusters based on position
        # NOTE: Ignore warning: Empty clusters (not associated with any point) are auto-resolved with random points
        w_all_pts = np.concatenate((
            self.wall_w_grid.reshape(-1, 2),
            self.wall_n_grid.reshape(-1, 2)))
        wall_pts = np.concatenate((
            self.wall_w_grid[wall_mask[:self.n_side_divs, :, cfg.SIDE_W_IDX]].reshape(-1, 2),
            self.wall_n_grid[wall_mask[:, :self.n_side_divs, cfg.SIDE_N_IDX]].reshape(-1, 2)))

        # Assign block colours
        b_all_pts = self.cell_pt_grid.reshape(-1, 2)
        link_pts = self.delim_pt_grid.reshape(-1, 2)

        cluster_centroids = (
            wall_pts
            if len(wall_pts) <= cfg.N_WALL_CLRS
            else kmeans2(wall_pts, cfg.N_WALL_CLRS, seed=self.rng)[0])

        closest_centroid_idcs = np.argmin(np.linalg.norm(w_all_pts[None] - cluster_centroids[:, None], axis=-1), axis=0)
        wall_clr_idcs = wall_clr_idx_map[closest_centroid_idcs]

        closest_centroid_idcs = np.argmin(np.linalg.norm(b_all_pts[None] - cluster_centroids[:, None], axis=-1), axis=0)
        block_clr_idcs = wall_clr_idx_map[closest_centroid_idcs]
        cell_clr_idx_grid = block_clr_idcs.reshape(block_mask.shape)

        closest_centroid_idcs = np.argmin(np.linalg.norm(link_pts[None] - cluster_centroids[:, None], axis=-1), axis=0)
        link_clr_idcs = wall_clr_idx_map[closest_centroid_idcs]

        # Uniformly assign bots to target objects
        bot_obj_map = np.arange(self.n_goals).repeat(ceil(self.n_bots / self.n_goals))[:self.n_bots]
        self.rng.shuffle(bot_obj_map)

        # Communication dropout
        speaker_mask = self.rng.random(self.n_bots) > self.speaker_dropout

        # Associate objects and bots with coloured goals
        obj_goal_map = np.sort(self.rng.permutation(cfg.N_GOAL_CLRS)[:self.n_goals])
        bot_goal_map = obj_goal_map[bot_obj_map]

        # Get map of connected cells
        cell_pass_map = self.init_connection_graph(wall_mask)

        # Randomise episode duration
        mul_min = 1. - cfg.MAX_DURATION_OFFSET
        mul_range = 2 * cfg.MAX_DURATION_OFFSET
        ep_duration = round(self.ep_duration * (mul_min + mul_range * self.rng.random()))

        return MazeData(
            spawn_pts=spawn_pts,
            spawn_angles=spawn_angles,
            speaker_mask=speaker_mask,
            decoy_pts=decoy_pts,
            obj_pts=obj_pts,
            bot_obj_map=bot_obj_map,
            obj_goal_map=obj_goal_map,
            bot_goal_map=bot_goal_map,
            free_cell_pts=free_cell_pts,
            target_cell_pts=target_cell_pts,
            free_cell_mask=free_cell_mask,
            target_cell_mask=target_cell_mask,
            spawn_cell_mask=self.spawn_cell_mask,
            wall_mask=wall_mask,
            block_mask=block_mask,
            link_mask=link_mask,
            link_clr_idcs=link_clr_idcs,
            cell_clr_idx_grid=cell_clr_idx_grid,
            wall_clr_idcs=wall_clr_idcs,
            wall_clr_idx_map=wall_clr_idx_map,
            cell_wall_grid=cell_wall_grid,
            cell_pass_map=cell_pass_map,
            global_spawn_flag=global_spawn_flag,
            ep_duration=ep_duration)

    # --------------------------------------------------------------------------
    # MARK: prune_grid

    def prune_grid(self, graph_edges: ndarray) -> 'tuple[ndarray, ndarray, ndarray, ndarray]':
        """Mask edges that intersect any edge of the given graph."""

        n_segments = self.n_side_divs
        n_delims = n_segments + 1

        # NOTE: [(0., 0.), (0., 0.)] is considered a null edge (clearing)
        cell_wall_grid = np.zeros((n_delims, n_delims, 2, 2, 2))

        link_mask = np.zeros((n_delims, n_delims), dtype=np.bool_)
        wall_mask = np.zeros((n_delims, n_delims, 2), dtype=np.bool_)

        # Iterate over pairs of edges in the grid
        for i in range(n_delims):
            for j in range(n_delims):

                # Check for intersections
                if i < n_segments:
                    ver_edge = self.delim_pt_grid[i:i+2, j]

                    if not any_intersections(*ver_edge, graph_edges):
                        link_mask[i:i+2, j] = True
                        wall_mask[i, j, cfg.SIDE_W_IDX] = True
                        cell_wall_grid[i, j, cfg.SIDE_W_IDX] = ver_edge

                if j < n_segments:
                    hor_edge = self.delim_pt_grid[i, j:j+2]

                    if not any_intersections(*hor_edge, graph_edges):
                        link_mask[i, j:j+2] = True
                        wall_mask[i, j, cfg.SIDE_N_IDX] = True
                        cell_wall_grid[i, j, cfg.SIDE_N_IDX] = hor_edge

        # Blocks are placed wherever all four underlying walls are raised
        block_mask = np.zeros((n_segments, n_segments), dtype=np.bool_)

        for i in range(n_segments):
            for j in range(n_segments):
                enclosed = (wall_mask[i:i+2, j, cfg.SIDE_N_IDX].sum() + wall_mask[i, j:j+2, cfg.SIDE_W_IDX].sum()) == 4

                if enclosed:
                    block_mask[i, j] = True

                elif self.level > 1:
                    if i > 0 and not block_mask[i-1, j]:
                        wall_mask[i, j, cfg.SIDE_N_IDX] = False
                        cell_wall_grid[i, j, cfg.SIDE_N_IDX] = 0.

                    if j > 0 and not block_mask[i, j-1]:
                        wall_mask[i, j, cfg.SIDE_W_IDX] = False
                        cell_wall_grid[i, j, cfg.SIDE_W_IDX] = 0.

        if self.level > 1:
            for i in range(1, n_segments):
                for j in range(1, n_segments):
                    link_mask[i, j] = block_mask[i-1:i+1, j-1:j+1].any()

        return link_mask, block_mask, wall_mask, cell_wall_grid

    # --------------------------------------------------------------------------
    # MARK: sample_viable_points

    def sample_viable_points(
        self,
        original_pts: ndarray,
        cell_wall_grid: ndarray,
        n_to_select: int,
        min_dist_to_same_pts: float = 0.,
        other_pts: ndarray = None,
        min_dist_to_other_pts: float = 0.,
        min_dist_to_wall_edges: float = 0.,
        max_dist_to_wall_edges: float = False,
        max_iterations: int = 1000
    ) -> 'None | ndarray':
        """
        Greedy point selection with point and edge constraints.
        If an iteration does not find `n_to_select` suitable points,
        selection can be retried with more points and in a different order.
        """

        max_offset = self.wall_halflength - min_dist_to_wall_edges
        candidate_pts = self.sample_around_points(original_pts, 2*n_to_select, max_offset)

        pts = np.zeros((n_to_select, 2))

        # Limit iterations to prevent needless cycling and overflow
        for _ in range(max_iterations):
            last_idx = 0

            for pt in candidate_pts:

                # Check distance to points of a different kind
                if (
                    (other_pts is not None) and
                    np.linalg.norm(pt - other_pts, axis=1).min() < min_dist_to_other_pts
                ):
                    continue

                # Check distance to already selected points
                if (
                    last_idx > 0 and
                    np.linalg.norm(pt - pts[:last_idx], axis=1).min() < min_dist_to_same_pts
                ):
                    continue

                # Check distance to edges
                cell_xidx, cell_yidx = np.digitize(pt, self.open_delims)
                cell_xidx = max(1, min(len(self.open_delims), cell_xidx))
                cell_yidx = max(1, min(len(self.open_delims), cell_yidx))

                edge_subset = cell_wall_grid[cell_xidx-1:cell_xidx+2, cell_yidx-1:cell_yidx+2].reshape(-1, 2, 2)

                min_dists_to_edges = [min_distance(*edge, pt) for edge in edge_subset]

                if any(dist < min_dist_to_wall_edges for dist in min_dists_to_edges):
                    continue

                elif max_dist_to_wall_edges and min(min_dists_to_edges) > max_dist_to_wall_edges:
                    continue

                # Assign selected point
                pts[last_idx] = pt
                last_idx += 1

                if last_idx == n_to_select:
                    return pts

            # On failure, expand and shuffle candidate pool
            candidate_pts = np.concatenate((
                candidate_pts,
                self.sample_around_points(original_pts, n_to_select, max_offset)))

            self.rng.shuffle(candidate_pts)

        raise TimeoutError(f'Bad config.: Could not select valid spawn points in {max_iterations} iterations.')

    # --------------------------------------------------------------------------
    # MARK: sample_around_points

    def sample_around_points(self, pts: ndarray, n_samples: int, max_offset: float) -> ndarray:
        """Sample points with replacement and add variance by random offsets."""

        return self.rng.choice(pts, n_samples, replace=True) + self.rng.uniform(-max_offset, max_offset, (n_samples, 2))

    # --------------------------------------------------------------------------
    # MARK: init_connection_graph

    def init_connection_graph(self, wall_mask: ndarray) -> 'dict[int, list[int]]':
        """Build a map between connected cells on the grid."""

        n_segments = self.n_side_divs

        pass_mask = ~wall_mask
        pass_w_mask, pass_n_mask = pass_mask[..., cfg.SIDE_W_IDX], pass_mask[..., cfg.SIDE_N_IDX]

        cell_pass_map = {i: [] for i in range(n_segments**2)}

        # Iterate over cell index coordinates
        for i, i_north, i_south in zip(range(n_segments), range(-1, n_segments-1), range(1, n_segments+1)):
            for j, j_east in zip(range(n_segments), range(1, n_segments+1)):

                # Check for passages
                pass_east = (j_east < n_segments) and pass_w_mask[i, j_east]
                pass_south = (i_south < n_segments) and pass_n_mask[i_south, j]
                pass_seast = pass_east and pass_south and pass_mask[i_south, j_east].all()
                pass_neast = pass_east and pass_n_mask[i, j] and pass_w_mask[i_north, j_east] and pass_n_mask[i, j_east]

                # Transform index coordinates to indices of a flattened array
                k = i * n_segments + j
                k_east = i * n_segments + j_east
                k_south = i_south * n_segments + j
                k_seast = i_south * n_segments + j_east
                k_neast = i_north * n_segments + j_east

                # Add entries to the map and edge list
                if pass_east:
                    cell_pass_map[k].append(k_east)
                    cell_pass_map[k_east].append(k)

                if pass_south:
                    cell_pass_map[k].append(k_south)
                    cell_pass_map[k_south].append(k)

                if pass_seast:
                    cell_pass_map[k].append(k_seast)
                    cell_pass_map[k_seast].append(k)

                if pass_neast:
                    cell_pass_map[k].append(k_neast)
                    cell_pass_map[k_neast].append(k)

        return cell_pass_map


# ------------------------------------------------------------------------------
# MARK: MazeValidator

class MazeValidator:
    """Test class for validating data, sampling, and path finding."""

    def __init__(self, sampler: MazeConstructor = None, data: MazeData = None):
        self.sampler = sampler

        # Sample a maze and init. the associated attributes
        self.data = data if data is not None else self.sampler.generate()

        # Extract walls
        wall_edges = self.data.cell_wall_grid.reshape(-1, 2, 2)
        self.wall_edges = wall_edges[np.any((wall_edges[:, 0, :] != wall_edges[:, 1, :]), axis=1)]
        self.wall_end_pts = np.unique(self.wall_edges.reshape(-1, 2), axis=0)

        # Locate passages
        self.cell_pts = self.sampler.cell_pt_grid.reshape(-1, 2)
        self.pass_edges = np.array([
            (self.cell_pts[i], self.cell_pts[j])
            for i, js in self.data.cell_pass_map.items() for j in js])

        # Convert passage map into numba typed dict.
        self.cellpair_path_map: 'dict[tuple[int, int], list[int]]' = get_numba_dict(tuple_as_key=True)
        self.cell_pass_map: 'dict[int, list[int]]' = get_numba_dict()

        for key, lst in self.data.cell_pass_map.items():
            self.cell_pass_map[key] = np.array(lst, dtype=np.int64)

        # Indices to colours
        self.obj_clrs = np.array(cfg.COLOURS['goal'])[self.data.obj_goal_map]
        self.border_clr = np.array(cfg.COLOURS['wall'])[self.data.wall_clr_idx_map[-1]]
        self.cell_clr_grid = np.array(cfg.COLOURS['wall'])[self.data.cell_clr_idx_grid]

    # --------------------------------------------------------------------------
    # MARK: get_info

    def get_info(self):
        print(f'Num. of walls ------------- {len(self.wall_edges)}')
        print(f'Num. of corners or seams -- {len(self.wall_end_pts)}')
        print(f'Num. of spawn points ------ {len(self.data.spawn_pts)}')
        print(f'Num. of target cells ------ {len(self.data.target_cell_pts)}')
        print(f'Num. of free cells -------- {len(self.data.free_cell_pts)}')
        print(f'Num. of cell connections -- {len(self.pass_edges)}')

    # --------------------------------------------------------------------------
    # MARK: get_path

    def get_sight(self, start_pt: ndarray, end_pt: ndarray) -> bool:
        entry_idcs = np.digitize(start_pt, self.sampler.open_delims)
        exit_idcs = np.digitize(end_pt, self.sampler.open_delims)

        return ray_trace(entry_idcs, exit_idcs, start_pt, end_pt, self.data.cell_wall_grid)

    def get_path(self, start_pt: ndarray, end_pt: ndarray) -> 'tuple[ndarray, float]':
        """Estimate the path and its length for a single test case."""

        entry_idcs = np.digitize(start_pt, self.sampler.open_delims)
        exit_idcs = np.digitize(end_pt, self.sampler.open_delims)

        entry_node = self.sampler.n_side_divs * entry_idcs[0] + entry_idcs[1]
        exit_node = self.sampler.n_side_divs * exit_idcs[0] + exit_idcs[1]

        in_sight = ray_trace(entry_idcs, exit_idcs, start_pt, end_pt, self.data.cell_wall_grid)

        dist = get_cached_paths(
                start_pt[None],
                end_pt[None],
                np.array([in_sight]),
                self.cell_pass_map,
                self.cellpair_path_map,
                self.sampler.open_delims,
                self.cell_pts,
                self.data.cell_wall_grid)[0][0]

        if in_sight:
            path_pts = np.array([end_pt, start_pt])

        else:
            path = prune_path_forward(
                self.cellpair_path_map[entry_node, exit_node],
                entry_idcs,
                start_pt,
                self.cell_pts,
                self.data.cell_wall_grid,
                self.sampler.n_side_divs)

            path_pts = np.concatenate((
                end_pt[None],
                self.cell_pts[path],
                start_pt[None]))

        return path_pts, dist
