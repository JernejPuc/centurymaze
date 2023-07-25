"""Core/numpy utilities"""

import os
from typing import Any, Callable

import numpy as np
import numba
from numpy import ndarray
from numba.core import types
from numba.typed import Dict
from scipy.spatial import Delaunay


def get_arg_defaults(fn: Callable) -> 'dict[str, Any]':
    """Get the input argument defaults of the given function."""

    varnames = fn.__code__.co_varnames
    defaults = fn.__defaults__

    if defaults is None:
        defaults = ()

    if len(defaults) < len(varnames):
        defaults = (None,) * (len(varnames) - len(defaults)) + defaults

    return {k: v for k, v in zip(varnames, defaults)}


def get_available_file_idx(data_dir: str, prefix: str) -> int:
    """Get the index of the first/next file with the given prefix."""

    filenames = os.listdir(data_dir)

    return 1 + max((
        int(name.split('_')[-1].split('.')[0])
        for name in filenames if name.startswith(prefix)), default=-1)


def urquhart(pts: ndarray) -> ndarray:
    """
    Form the (cyclical, connected) Urquhart graph from the Delaunay
    triangulation of given points.
    """

    tri = Delaunay(pts)

    # Get unique edges in the triangulation
    edges = set()

    for i, j, k in tri.simplices:
        edges.add(tuple(sorted((i, j))))
        edges.add(tuple(sorted((i, k))))
        edges.add(tuple(sorted((j, k))))

    # Remove the longest edge from each triangle in the triangulation
    for simplex in tri.simplices:
        i, j, k = simplex
        pi, pj, pk = pts[simplex]

        argmax = np.argmax(np.linalg.norm((pi - pj, pi - pk, pj - pk), axis=1))

        if argmax == 0:
            u, v = i, j

        elif argmax == 1:
            u, v = i, k

        else:
            u, v = j, k

        try:
            edges.remove(tuple(sorted((u, v))))

        except KeyError:
            pass

    # Return remaining edges as pairs of coordinates
    return np.array([[pts[idx] for idx in edge] for edge in edges])


def any_intersections(p1: ndarray, p2: ndarray, edges: ndarray) -> bool:
    """Check for intersections between a segment and a set of edges."""

    return any(intersection(p1, p2, q1, q2) for q1, q2 in edges)


@numba.jit(nopython=True, nogil=True, cache=True)
def intersection(p1: ndarray, p2: ndarray, q1: ndarray, q2: ndarray) -> bool:
    """Check for an intersection between two line segments."""

    # Null edge case
    if q1[0] == q2[0] and q1[1] == q2[1]:
        return False

    o1 = orientation(p1, p2, q1)
    o2 = orientation(p1, p2, q2)
    o3 = orientation(q1, q2, p1)
    o4 = orientation(q1, q2, p2)

    return (
        (o1 != o2 and o3 != o4) or
        (o1 == 0 and on_segment(p1, p2, q1)) or
        (o2 == 0 and on_segment(p1, p2, q2)) or
        (o3 == 0 and on_segment(q1, q2, p1)) or
        (o4 == 0 and on_segment(q1, q2, p2)))


@numba.jit(nopython=True, nogil=True, cache=True)
def orientation(p: ndarray, q: ndarray, r: ndarray) -> int:
    """Get the orientation (order class) of a pqr point triplet."""

    val = ((q[1] - p[1]) * (r[0] - q[0])) - ((q[0] - p[0]) * (r[1] - q[1]))

    return 1 if val > 0 else (2 if val < 0 else 0)


@numba.jit(nopython=True, nogil=True, cache=True)
def on_segment(p: ndarray, q: ndarray, r: ndarray) -> bool:
    """Check if point r lies on the segment of points pq."""

    return (
        min(p[0], q[0]) <= r[0] <= max(p[0], q[0]) and
        min(p[1], q[1]) <= r[1] <= max(p[1], q[1]))


@numba.jit(nopython=True, nogil=True, cache=True)
def min_distance(p: ndarray, q: ndarray, r: ndarray) -> float:
    """Get the distance of point r to the segment of points pq."""

    # Null edge case
    if p[0] == q[0] and p[1] == q[1]:
        return np.inf

    pq = q - p
    qr = r - q
    pr = r - p

    if np.dot(pq, qr) > 0.:
        return np.linalg.norm(qr)

    elif np.dot(pq, pr) < 0.:
        return np.linalg.norm(pr)

    return abs(pq[0]*pr[1] - pq[1]*pr[0]) / np.linalg.norm(pq)


@numba.jit(nopython=True, nogil=True, cache=True)
def ray_trace(
    i0: ndarray,
    i1: ndarray,
    p0: ndarray,
    p1: ndarray,
    grid_edge_pairs: ndarray
) -> bool:
    """Bresenham line tracer checking for possible intersections on the go."""

    x0, y0 = i0[0], i0[1]
    x1, y1 = i1[0], i1[1]

    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)

    e = dx + dy

    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1

    while True:
        for i in range(2):
            if intersection(p0, p1, grid_edge_pairs[x0, y0, i, 0], grid_edge_pairs[x0, y0, i, 1]):
                return False

        if x0 == x1 and y0 == y1:
            break

        y0_ = y0 + sy
        x0_ = x0 + sx

        if y0 != y1:
            for i in range(2):
                if intersection(p0, p1, grid_edge_pairs[x0, y0_, i, 0], grid_edge_pairs[x0, y0_, i, 1]):
                    return False

        if x0 != x1:
            for i in range(2):
                if intersection(p0, p1, grid_edge_pairs[x0_, y0, i, 0], grid_edge_pairs[x0_, y0, i, 1]):
                    return False

        e2 = 2*e

        if e2 >= dy:
            e += dy
            x0 = x0_

        if e2 <= dx:
            e += dx
            y0 = y0_

    return True


@numba.jit(nopython=True, nogil=True, cache=True)
def eval_line_of_sight(
    origin_pos: ndarray,
    target_pos: ndarray,
    target_idcs: ndarray,
    open_grid_delims: ndarray,
    grid_edge_pair_list: ndarray
) -> ndarray:
    """Run ray tracing from multiple origins to their targets."""

    origin_idcs = np.digitize(origin_pos, open_grid_delims)
    grid_step = len(origin_pos) // len(grid_edge_pair_list)

    return np.array([
        ray_trace(
            origin_idcs[i],
            target_idcs[i],
            origin_pos[i],
            target_pos[i],
            grid_edge_pair_list[i // grid_step])
        for i in range(len(origin_pos))])


def get_numba_dict(tuple_as_key: bool = False) -> 'dict[int | tuple[int, int], ndarray]':
    """Create a typed dictionary for use in njit functions."""

    key_type = types.UniTuple(types.int64, 2) if tuple_as_key else types.int64

    return Dict.empty(key_type=key_type, value_type=types.int64[:])


@numba.jit(nopython=True, nogil=True, cache=True)
def astar(
    graph: 'dict[int, ndarray]',
    node_pos: ndarray,
    entry_node: ndarray,
    exit_node: ndarray
) -> ndarray:
    """A* with air distance heuristic returning a reversed path."""

    entry_pos = node_pos[entry_node]
    exit_pos = node_pos[exit_node]

    # If no path can be found, use air path estimate
    if len(graph) == 0:
        return np.array([exit_node, entry_node])

    # Set of discovered nodes (impl. as dict due to a persisting bug)
    # See: https://github.com/numba/numba/issues/8627)
    candidate_set = Dict.empty(key_type=types.int64, value_type=types.int64)
    candidate_set[entry_node] = 0

    # Map of immediately preceding node on the cheapest path to key
    preceders = Dict.empty(key_type=types.int64, value_type=types.int64)

    # Map of the cheapest path cost to key
    g_scores = {entry_node: 0.}

    # Map of the estimated cost for the cheapest path through key (default inf)
    f_scores = {entry_node: np.linalg.norm(entry_pos - exit_pos)}

    # Argmin should always exist, this just avoids it looking unbound
    current_node = entry_node

    # Repeat until exit is reached or nodes are exhausted
    while len(candidate_set) > 0:

        # Find node in candidate_set with the lowest f_score
        min_score = np.inf

        for key in candidate_set:
            val = f_scores[key]

            if val < min_score:
                min_score = val
                current_node = key

        # Check terminal condition and reconstruct the path
        if current_node == exit_node:
            path = [current_node]

            while current_node in preceders:
                current_node = preceders[current_node]
                path.append(current_node)

            return np.array(path)

        # Update neighbours
        del candidate_set[current_node]
        current_pos = node_pos[current_node]
        current_g_score = g_scores[current_node]

        for neigh_node in graph[current_node]:
            neigh_pos = node_pos[neigh_node]

            old_g_score = g_scores[neigh_node] if neigh_node in g_scores else np.inf
            new_g_score = current_g_score + np.linalg.norm(current_pos - neigh_pos)

            if new_g_score < old_g_score:
                g_scores[neigh_node] = new_g_score
                f_scores[neigh_node] = new_g_score + np.linalg.norm(neigh_pos - exit_pos)
                preceders[neigh_node] = current_node

                candidate_set[neigh_node] = 0

    # If no path can be found, use air path estimate
    return np.array([exit_node, entry_node])


@numba.jit(nopython=True, nogil=True, cache=True)
def reconstruct_paths(
    origin_pos: ndarray,
    target_pos: ndarray,
    sight_mask: ndarray,
    path_map: 'dict[tuple[int, int], ndarray]',
    open_grid_delims: ndarray,
    grid_square_centres: ndarray,
    grid_edge_pairs: ndarray
) -> 'tuple[ndarray, ndarray]':
    """
    Estimate path length and starting direction from valid starting points
    to target end points based on precomputed A* reference paths.
    """

    # Compute air distance to goal
    diffs = target_pos - origin_pos
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    dirs = diffs / np.expand_dims(dists, 1)

    # Check for points without their goal in sight
    remaining_pt_idcs = np.where(~sight_mask)[0]

    if len(remaining_pt_idcs) == 0:
        return dists, dirs

    # For remaining points, find entry and exit nodes (bounding squares)
    origin_pos = origin_pos[remaining_pt_idcs]
    target_pos = target_pos[remaining_pt_idcs]

    entry_idcs = np.digitize(origin_pos, open_grid_delims)
    exit_idcs = np.digitize(target_pos, open_grid_delims)

    # Node IDs are flattened grid indices
    n_squares_per_row = len(open_grid_delims) + 1
    entry_nodes = n_squares_per_row * entry_idcs[:, 0] + entry_idcs[:, 1]
    exit_nodes = n_squares_per_row * exit_idcs[:, 0] + exit_idcs[:, 1]

    # Traverse graph for each remaining point until their goal is in sight
    for i, pt_i in enumerate(remaining_pt_idcs):
        path = path_map[entry_nodes[i], exit_nodes[i]]

        path = prune_path_forward(
            path,
            entry_idcs[i],
            origin_pos[i],
            grid_square_centres,
            grid_edge_pairs,
            n_squares_per_row)

        dists[pt_i], dirs[pt_i] = eval_path(
            path,
            origin_pos[i],
            target_pos[i],
            grid_square_centres)

    return dists, dirs


@numba.jit(nopython=True, nogil=True, cache=True)
def eval_path(
    path: ndarray,
    origin_pos: ndarray,
    target_pos: ndarray,
    grid_square_centres: ndarray
) -> 'tuple[float, ndarray]':
    """
    Sum the distances between consecutive points on a path from the origin
    to the target and get a starting direction.
    """

    # Convert to coordinates
    path_pts = grid_square_centres[path]

    # Distances between points on the path
    if len(path) == 1:
        dist_sum = 0.

    else:
        diffs = path_pts[:-1] - path_pts[1:]
        dist_sum = np.sum(np.sqrt(np.sum(diffs**2, axis=1)))

    # Starting direction
    diff_to_next = path_pts[-1] - origin_pos
    dir_to_next = diff_to_next / np.linalg.norm(diff_to_next)

    # Distances wrt. entry and exit nodes
    dist_sum += np.linalg.norm(diff_to_next) + np.linalg.norm(target_pos - path_pts[0])

    return dist_sum, dir_to_next


@numba.jit(nopython=True, nogil=True, cache=True)
def prune_path_forward(
    path: ndarray,
    entry_idcs: ndarray,
    origin_pos: ndarray,
    grid_square_centres: ndarray,
    grid_edge_squares: ndarray,
    n_squares_per_row: int
) -> ndarray:
    """
    Once a path is constructed, a line can be traced from the origin
    to succeeding points and they are removed if sight is established,
    effectively shortening the path.
    """

    entry_ptr = len(path)-1

    node_idcs = np.array([0, 0], dtype=np.int64)

    for ptr in range(len(path)-2, -1, -1):
        node_idx = path[ptr]
        node_idcs[0] = node_idx // n_squares_per_row
        node_idcs[1] = node_idx % n_squares_per_row
        node_pos = grid_square_centres[node_idx]

        if ray_trace(entry_idcs, node_idcs, origin_pos, node_pos, grid_edge_squares):
            entry_ptr = ptr

        else:
            break

    return path[:entry_ptr+1]


@numba.jit(nopython=True, nogil=True, cache=True)
def prune_path_backward(
    path: ndarray,
    exit_idcs: ndarray,
    target_pos: ndarray,
    grid_square_centres: ndarray,
    grid_edge_squares: ndarray,
    n_squares_per_row: int
) -> ndarray:
    """
    Once a path is constructed, a line can be traced from the target
    to preceding points and they are removed if sight is established,
    effectively shortening the path.
    """

    exit_ptr = 0

    node_idcs = np.array([0, 0], dtype=np.int64)

    for ptr in range(1, len(path)):
        node_idx = path[ptr]
        node_idcs[0] = node_idx // n_squares_per_row
        node_idcs[1] = node_idx % n_squares_per_row
        node_pos = grid_square_centres[node_idx]

        if ray_trace(exit_idcs, node_idcs, target_pos, node_pos, grid_edge_squares):
            exit_ptr = ptr

        else:
            break

    return path[exit_ptr:]
