"""Maze initialisation"""

import json
from typing import Any

import numpy as np
from numpy import ndarray

import config as cfg
from utils import (
    get_cached_paths,
    get_numba_dict,
    min_distance,
    prune_path_forward,
    ray_trace)


# ------------------------------------------------------------------------------
# MARK: MazeData

class MazeData:
    """Reads the topology and functional elements from maze descriptor files."""

    spawn_pts: ndarray
    target_cell_pts: ndarray
    free_cell_pts: ndarray
    cell_pt_grid: ndarray
    cell_wall_grid: ndarray
    cell_pass_map: 'dict[int, ndarray]'

    spawn_cell_mask: ndarray
    target_cell_mask: ndarray
    free_cell_mask: ndarray
    cell_clr_idx_grid: ndarray
    wall_mask: ndarray
    wall_clr_idx_grid: ndarray

    def __init__(self, partial_path: str):
        self.save_path = partial_path + '_data.npz'

        with open(partial_path + '_spec.json', 'r') as f:
            self.spec = json.load(f)

        with open(partial_path + '_plan.txt', 'r') as f:
            self.plan = f.read()

        self.length = self.spec['grid']['side_length']
        self.n_divs = self.spec['grid']['n_side_divs']
        self.n_sub_divs = self.spec['grid']['n_cell_divs']

        self.n_delims = self.n_divs + 1
        self.delims = np.linspace(self.length/-2, self.length/2, self.n_delims)
        self.open_delims = self.delims[1:-1]

        midpts = np.round((self.delims[1:] + self.delims[:-1]) / 2, 2)
        self.cell_pt_grid = np.stack(np.meshgrid(midpts, midpts, indexing='ij'), axis=-1)

        self.init_cell_states()
        self.init_candidate_points()
        self.init_connection_graph()

    # --------------------------------------------------------------------------
    # MARK: init_cell_states

    def init_cell_states(self):
        """Fill grids and masks for cell walls, colours, and function."""

        self.spawn_cell_mask = np.zeros((self.n_divs, self.n_divs), dtype=np.bool_)
        self.target_cell_mask = np.ones((self.n_divs, self.n_divs), dtype=np.bool_)
        self.free_cell_mask = np.ones((self.n_divs, self.n_divs), dtype=np.bool_)
        self.wall_mask = np.zeros((self.n_delims, self.n_delims, 2), dtype=np.bool_)

        self.cell_wall_grid = np.zeros((self.n_delims, self.n_delims, 2, 2, 2))
        self.cell_clr_idx_grid = np.ones((self.n_divs, self.n_divs), dtype=np.int64) * -1
        self.wall_clr_idx_grid = np.ones((self.n_delims, self.n_delims, 2), dtype=np.int64) * -1

        plan_lines = self.plan.splitlines()
        main_lines, clr_lines = plan_lines[:self.n_delims], plan_lines[self.n_delims+1:self.n_delims*2+1]

        # Iterate over cell markers
        for i, main_line, clr_line in zip(range(-1, self.n_delims-1), main_lines, clr_lines):
            i_w, i_n = i, i+1

            for j, j3 in enumerate(range(0, 3*self.n_delims, 3)):
                wall_x, wall_y, cell_func = main_line[j3:j3+3]
                clr_x, clr_y = clr_line[j3:j3+2]

                # X-spanning wall (N -> S, side W)
                if wall_x == '|':
                    self.wall_mask[i_w, j, cfg.SIDE_W_IDX] = True
                    self.wall_clr_idx_grid[i_w, j, cfg.SIDE_W_IDX] = int(clr_x)
                    self.cell_wall_grid[i_w, j, cfg.SIDE_W_IDX] = self.delims[[i_w, j]], self.delims[[i_w+1, j]]

                # Y-spanning wall (W -> E, side N)
                if wall_y == '_':
                    self.wall_mask[i_n, j, cfg.SIDE_N_IDX] = True
                    self.wall_clr_idx_grid[i_n, j, cfg.SIDE_N_IDX] = int(clr_y)
                    self.cell_wall_grid[i_n, j, cfg.SIDE_N_IDX] = self.delims[[i_n, j]], self.delims[[i_n, j+1]]

                # Candidate spawn cell
                if cell_func == '^':
                    self.spawn_cell_mask[i, j] = True
                    self.target_cell_mask[i, j] = False

                # Prohibited target cell
                elif cell_func == '.':
                    self.target_cell_mask[i, j] = False

        # Detect cell enclosures
        for i in range(self.n_divs):
            for j in range(self.n_divs):
                not_enclosed = \
                    self.wall_mask[i, j:j+2, cfg.SIDE_W_IDX].sum() + self.wall_mask[i:i+2, j, cfg.SIDE_N_IDX].sum() != 4

                if not_enclosed:
                    continue

                self.free_cell_mask[i, j] = False

                clr_freq = np.bincount(
                    np.concatenate((
                        self.wall_clr_idx_grid[i, j:j+2, cfg.SIDE_W_IDX],
                        self.wall_clr_idx_grid[i:i+2, j, cfg.SIDE_N_IDX])),
                    minlength=len(cfg.COLOURS['wall']))

                self.cell_clr_idx_grid[i, j] = np.argmax(clr_freq)

        self.target_cell_mask &= self.free_cell_mask

    # --------------------------------------------------------------------------
    # MARK: init_spawn_points

    def init_candidate_points(self):
        """Extract points of cells to be used for spawning or as targets."""

        cell_spacing = self.length / self.n_divs
        cell_sub_spacing = self.length / (self.n_sub_divs * self.n_divs)

        offsets = np.linspace(-1., 1., self.n_sub_divs) * (cell_spacing - cell_sub_spacing) / 2.
        offsets = np.stack(np.meshgrid(offsets, offsets, indexing='ij'), axis=-1).reshape(-1, 1, 2)

        spawn_cell_pts = self.cell_pt_grid[self.spawn_cell_mask].reshape(1, -1, 2)
        self.spawn_pts = (spawn_cell_pts + offsets).reshape(-1, 2)

        spawn_pt_centroid = spawn_cell_pts.mean(axis=1)
        spawn_pt_dists = np.linalg.norm(self.spawn_pts - spawn_pt_centroid, axis=-1)
        self.spawn_pts = self.spawn_pts[np.argsort(spawn_pt_dists)]

        self.target_cell_pts = self.cell_pt_grid[self.target_cell_mask].reshape(-1, 2)
        self.free_cell_pts = self.cell_pt_grid[self.free_cell_mask].reshape(-1, 2)

    # --------------------------------------------------------------------------
    # MARK: init_connection_graph

    def init_connection_graph(self):
        """Build a map between connected cells on the grid."""

        pass_mask = ~self.wall_mask
        pass_w_mask, pass_n_mask = pass_mask[..., cfg.SIDE_W_IDX], pass_mask[..., cfg.SIDE_N_IDX]

        self.cell_pass_map = {i: [] for i in range(self.n_divs**2)}

        # Iterate over cell index coordinates
        for i, i_north, i_south in zip(range(self.n_divs), range(-1, self.n_divs-1), range(1, self.n_divs+1)):
            for j, j_east in zip(range(self.n_divs), range(1, self.n_divs+1)):

                # Check for passages
                pass_east = (j_east < self.n_divs) and pass_w_mask[i, j_east]
                pass_south = (i_south < self.n_divs) and pass_n_mask[i_south, j]
                pass_seast = pass_east and pass_south and pass_mask[i_south, j_east].all()
                pass_neast = pass_east and pass_n_mask[i, j] and pass_w_mask[i_north, j_east] and pass_n_mask[i, j_east]

                # Transform index coordinates to indices of a flattened array
                k = i * self.n_divs + j
                k_east = i * self.n_divs + j_east
                k_south = i_south * self.n_divs + j
                k_seast = i_south * self.n_divs + j_east
                k_neast = i_north * self.n_divs + j_east

                # Add entries to the map and edge list
                if pass_east:
                    self.cell_pass_map[k].append(k_east)
                    self.cell_pass_map[k_east].append(k)

                if pass_south:
                    self.cell_pass_map[k].append(k_south)
                    self.cell_pass_map[k_south].append(k)

                if pass_seast:
                    self.cell_pass_map[k].append(k_seast)
                    self.cell_pass_map[k_seast].append(k)

                if pass_neast:
                    self.cell_pass_map[k].append(k_neast)
                    self.cell_pass_map[k_neast].append(k)

    # --------------------------------------------------------------------------
    # MARK: save

    def save(self):
        """Save the dicts. and arrays needed to set up the maze in simulation."""

        np.savez(
            self.save_path,
            spec=self.spec,
            spawn_pts=self.spawn_pts,
            target_cell_pts=self.target_cell_pts,
            free_cell_pts=self.free_cell_pts,
            open_delims=self.delims[1:-1],
            cell_pt_grid=self.cell_pt_grid,
            cell_wall_grid=self.cell_wall_grid,
            cell_clr_idx_grid=self.cell_clr_idx_grid,
            cell_pass_map=self.cell_pass_map)


# --------------------------------------------------------------------------
# MARK: MazeSampler

class MazeSampler:
    """Samples object and bot spawn states and distributes goals."""

    def __init__(
        self,
        data: 'MazeData | np.lib.npyio.NpzFile',
        n_bots: int,
        n_goals: int = None,
        global_spawn_prob: float = 0.,
        rng: 'None | int | np.random.Generator' = None
    ):
        if isinstance(data, MazeData):
            spec = data.spec
            data = data.__dict__

        else:
            spec = data['spec'].item()

        self.spawn_pts = data['spawn_pts']
        self.target_cell_pts = data['target_cell_pts']
        self.free_cell_pts = data['free_cell_pts']
        self.cell_wall_grid = data['cell_wall_grid']
        self.open_delims = data['open_delims']

        self.wall_halflength = spec['grid']['side_length'] / (2 * spec['grid']['n_side_divs'])
        self.min_goal_spacing = spec['grid']['side_length'] / 3

        self.n_bots = n_bots
        self.n_goals = n_goals if n_goals is not None else cfg.N_GOAL_CLRS
        self.global_spawn_prob = global_spawn_prob
        self.rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

    # --------------------------------------------------------------------------
    # MARK: sample_tasks

    def sample_tasks(self) -> 'tuple[ndarray, ...]':
        """Select starting points, angles, and goal distribution."""

        obj_pts = self.sample_viable_points(
                self.target_cell_pts,
                self.n_goals, self.min_goal_spacing,
                None, 0.,
                cfg.OBJ_TO_WALL_GAP)

        # Option to spawn out of the designated area for random data collection
        if self.global_spawn_prob and self.rng.random() < self.global_spawn_prob:
            spawn_pts = self.sample_viable_points(
                self.free_cell_pts,
                self.n_bots, cfg.BOT_TO_BOT_GAP,
                obj_pts, cfg.BOT_TO_OBJ_GAP,
                cfg.BOT_TO_WALL_GAP)

        # Otherwise use pre-sorted points closest to the centroid of that area
        else:
            spawn_pts = self.spawn_pts[:self.n_bots]

        spawn_angles = self.rng.uniform(low=-np.pi, high=np.pi, size=self.n_bots)

        # Uniformly assign bots to target objects
        bot_obj_map = np.arange(self.n_goals).repeat(int(np.ceil(self.n_bots / self.n_goals)))[:self.n_bots]
        self.rng.shuffle(bot_obj_map)

        # Associate objects and bots with coloured goals
        obj_goal_map = np.sort(self.rng.permutation(cfg.N_GOAL_CLRS)[:self.n_goals])
        bot_goal_map = obj_goal_map[bot_obj_map]

        return obj_pts, spawn_pts, spawn_angles, bot_obj_map, obj_goal_map, bot_goal_map

    # --------------------------------------------------------------------------
    # MARK: sample_viable_points

    def sample_viable_points(
        self,
        original_pts: ndarray,
        n_to_select: int,
        min_dist_to_same_pts: float = 0.,
        other_pts: ndarray = None,
        min_dist_to_other_pts: float = 0.,
        min_dist_to_wall_edges: float = 0.,
        max_iterations: int = 100
    ) -> ndarray:
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

                edge_subset = self.cell_wall_grid[cell_xidx-1:cell_xidx+2, cell_yidx-1:cell_yidx+2].reshape(-1, 2, 2)

                if any(min_distance(*edge, pt) < min_dist_to_wall_edges for edge in edge_subset):
                    continue

                # Assign selected point
                pts[last_idx] = pt
                last_idx += 1

                if last_idx == n_to_select:
                    return pts

            # Expand and shuffle candidate pool
            candidate_pts = np.concatenate((
                candidate_pts,
                self.sample_around_points(original_pts, n_to_select, max_offset)))

            self.rng.shuffle(candidate_pts)

        raise RuntimeError('Bad configuration; maximum number of iterations exceeded.')

    # --------------------------------------------------------------------------
    # MARK: sample_around_points

    def sample_around_points(self, pts: ndarray, n_samples: int, max_offset: float) -> ndarray:
        """Sample points with replacement and add variance by random offsets."""

        return self.rng.choice(pts, n_samples, replace=True) + self.rng.uniform(-max_offset, max_offset, (n_samples, 2))


# ------------------------------------------------------------------------------
# MARK: MazeValidator

class MazeValidator:
    """Test class for validating data, sampling, and path finding."""

    cellpair_path_map: 'dict[tuple[int, int], list[int]]'

    obj_pts: ndarray
    spawn_pts: ndarray
    bot_obj_map: ndarray
    obj_goal_map: ndarray
    obj_clrs: ndarray

    def __init__(self, data: MazeData, sampler: MazeSampler = None):
        self.data = data
        self.sampler = sampler

        # Extract walls
        wall_edges = data.cell_wall_grid.reshape(-1, 2, 2)
        self.wall_edges = wall_edges[np.any((wall_edges[:, 0, :] != wall_edges[:, 1, :]), axis=1)]
        self.wall_end_pts = np.unique(self.wall_edges.reshape(-1, 2), axis=0)

        # Locate passages
        self.cell_pts = data.cell_pt_grid.reshape(-1, 2)
        self.pass_edges = np.array([
            (self.cell_pts[i], self.cell_pts[j])
            for i, js in data.cell_pass_map.items() for j in js])

        # Convert passage map into numba typed dict.
        self.cell_pass_map = get_numba_dict()

        for key, lst in data.cell_pass_map.items():
            self.cell_pass_map[key] = np.array(lst, dtype=np.int64)

        # Indices to colours
        self.wall_clr_grid = np.array(cfg.COLOURS['wall'])[data.wall_clr_idx_grid]
        self.cell_clr_grid = np.array(cfg.COLOURS['wall'])[data.cell_clr_idx_grid]

    def __setattr__(self, name: str, value: 'MazeSampler | Any'):
        """Wrapper to automatically reset sampled state."""

        object.__setattr__(self, name, value)

        if name == 'sampler' and isinstance(value, MazeSampler):
            self.reset()

    # --------------------------------------------------------------------------
    # MARK: reset

    def reset(self):
        """Reinit. the path map, points, and goal distribution."""

        self.cellpair_path_map = get_numba_dict(tuple_as_key=True)

        self.obj_pts, self.spawn_pts, _, self.bot_obj_map, self.obj_goal_map, _ = self.sampler.sample_tasks()
        self.obj_clrs = np.array(cfg.COLOURS['goal'])[self.obj_goal_map]

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

        entry_node = self.data.n_divs * entry_idcs[0] + entry_idcs[1]
        exit_node = self.data.n_divs * exit_idcs[0] + exit_idcs[1]

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
            path = np.array([exit_node, entry_node])

        else:
            path = prune_path_forward(
                self.cellpair_path_map[entry_node, exit_node],
                entry_idcs,
                start_pt,
                self.cell_pts,
                self.data.cell_wall_grid,
                self.data.n_divs)

        path_pts = self.cell_pts[path]

        return path_pts, dist
