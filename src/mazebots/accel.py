"""Operations accelerated with CUDA graphs"""

from typing import Callable

import torch
from torch import cuda, Tensor

from discit.accel import capture_graph

from utils_torch import adjust_depth_range, apply_quat_rot, get_eulz_from_quat, MAX_DIST


# Mag. units are nT / 100
# Data for 46-3-5N (46.0513889) 14-30-22E 295M on 2023-03-22
# https://www.ngdc.noaa.gov/geomag/calculators/magcalc.shtml#igrfwmm
# https://www.sensorsone.com/local-gravity-calculator/
MAG_REF = [[0.220967, 0.017385, 0.428883]]
ACC_REF = [[0., 0., -0.980624]]

# RGB receiver phases
_RAD_120 = 120./180. * torch.pi
RCVR_REL_PHASES = [[0., -_RAD_120, _RAD_120]]
_2PI = 2.*torch.pi

# Limited to 10 in the worst case, but usually it should be much lower
# Implemented with tanh(x/10)*10 to keep sensitivity in the average case
SCALE_RCVR = 10.
SCALE_ACC = 10.

# https://forums.developer.nvidia.com/t/inaccuracy-of-dof-state-readings/197373
SCALE_DOF = 1./20.

# Expecting at most half of a turn per second
SCALE_ANG_VEL = 1./torch.pi

# Expecting 1 avg. completion per 3 to 60 seconds, i.e. 3 to 0.17 per 10 seconds
SCALE_THROUGHPUT = 10.

# Episodes last from 8 to 600 seconds, i.e. 0.133 to 10 minutes, dt is 0.0042 min
SCALE_TIME = 1./60.

# Task completion confirmed after 1 second at goal
T_TASK_CONFIRM = 1.


class SessionTensorSignature:
    actor_states: Tensor
    dof_vel: Tensor
    bot_pos: Tensor
    bot_ori: Tensor
    bot_vel: Tensor
    bot_ang_vel: Tensor
    bot_old_vel: Tensor
    img_rgb_list: 'list[Tensor]'
    img_dep_list: 'list[Tensor]'
    img_seg_list: 'list[Tensor]'
    obj_trans_prob: Tensor
    obj_rgb: Tensor
    obj_pos: Tensor
    goal_rgb: Tensor
    goal_pos: Tensor
    goal_path_len: Tensor
    goal_path_dir: Tensor
    goal_in_sight_mask: Tensor
    env_run_times: Tensor
    bot_time_on_task: Tensor
    bot_time_at_goal: Tensor
    throughput: Tensor
    bot_done_ctr: Tensor
    bot_done_mask: Tensor
    bot_rst_mask: Tensor
    bot_pos: Tensor
    bot_vel: Tensor
    act_trq: Tensor
    act_rgb: Tensor


class OpAccelerator:
    """
    Converts and exposes a subset of ops for optimised execution with CUDA graphs.
    Speedups of about 4x are expected.

    NOTE: Sharing memory pools can cause problems, but sharing IO is fine.
    """

    def __init__(
        self,
        session: SessionTensorSignature,
        n_envs: int,
        n_bots: int,
        ep_duration: int,
        steps_per_second: float,
        objective_radius: float,
        device: str
    ):
        self.n_envs = n_envs
        self.n_bots = n_bots
        self.n_all_bots = n_envs * n_bots
        self.dt = 1. / steps_per_second
        self.objective_radius = objective_radius

        # Init on device
        self.ep_duration = torch.tensor(ep_duration, dtype=torch.float32, device=device)
        self.mag_ref = torch.tensor(MAG_REF, dtype=torch.float32, device=device)
        self.acc_ref = torch.tensor(ACC_REF, dtype=torch.float32, device=device)
        self.quat_inv = torch.tensor([[-1., -1., -1., 1.]], dtype=torch.float32, device=device)
        self.rcvr_rel_phases = torch.tensor(RCVR_REL_PHASES, dtype=torch.float32, device=device)
        self.zero_z = torch.zeros((self.n_all_bots, 1), dtype=torch.float32, device=device)

        self.graphs: 'dict[str, dict[str, tuple[Tensor, ...] | cuda.CUDAGraph | Callable]]' = {}

        # Graph 1
        inputs = (
            session.bot_pos,
            session.bot_ori,
            session.goal_pos,
            session.goal_path_len,
            session.goal_path_dir,
            session.goal_in_sight_mask,
            session.env_run_times,
            session.bot_time_on_task,
            session.bot_time_at_goal,
            session.bot_done_ctr)

        self.eval_state, self.graphs['eval_state'] = \
            capture_graph(self.eval_state, inputs, copy_idcs_in=(), copy_idcs_out=(8,))  # Copy reward

        # Reset tensors modified in-place during warm-up and capture
        session.env_run_times.zero_()
        session.bot_time_on_task.zero_()
        session.bot_time_at_goal.zero_()
        session.bot_done_ctr.zero_()

        # Graph 2
        inputs = (
            session.bot_pos,
            session.bot_vel,
            session.bot_old_vel,
            session.bot_ori,
            session.bot_ang_vel,
            session.dof_vel,
            session.act_trq,    # Copy
            session.act_rgb,    # Copy
            session.goal_rgb,
            session.bot_time_at_goal,
            self.graphs['eval_state']['out'][3])  # Throughput

        self.get_vector_observations, self.graphs['get_vector_observations'] = \
            capture_graph(self.get_vector_observations, inputs, copy_idcs_in=(6, 7), copy_idcs_out=())

        # Graph 3
        inputs = (session.img_rgb_list, session.img_dep_list)

        self.get_image_observations, self.graphs['get_image_observations'] = \
            capture_graph(self.get_image_observations, inputs, copy_idcs_in=(), copy_idcs_out=())

    def accel_action(
        self,
        act_fn: Callable,
        aux_tensors: 'tuple[Tensor, ...]' = None,
        suffix: str = ''
    ) -> Callable:

        if aux_tensors is None:
            aux_tensors = ()

        # Graph 4
        inputs = (
            self.graphs['get_image_observations']['out'],
            self.graphs['get_vector_observations']['out'],
            self.graphs['eval_state']['out'][-1],
            *[aux.detach().clone() for aux in aux_tensors])

        act_fn, self.graphs[f'act_{suffix}' if suffix else 'act'] = \
            capture_graph(act_fn, inputs, copy_idcs_in=tuple(range(3, 3+len(aux_tensors))))

        return act_fn

    def eval_state(
        self,
        bot_pos: Tensor,
        bot_ori: Tensor,
        goal_pos: Tensor,
        goal_path_len: Tensor,
        goal_path_dir: Tensor,
        goal_in_sight_mask: Tensor,
        env_run_times: Tensor,
        bot_time_on_task: Tensor,
        bot_time_at_goal: Tensor,
        bot_done_ctr: Tensor
    ):
        # Get distance and direction to objectives
        goal_diff = goal_pos - bot_pos
        goal_dist: Tensor = torch.linalg.norm(goal_diff, dim=1)
        goal_dir = goal_diff / goal_dist[..., None]

        # Check if any goals are in reach
        goal_in_reach_mask = (goal_dist < self.objective_radius) & goal_in_sight_mask

        # Step time
        # NOTE: In-place ops avoid needless copying between graph outputs and inputs
        env_run_times = env_run_times.add_(self.dt)
        bot_time_on_task = bot_time_on_task.add_(self.dt)
        bot_time_at_goal = bot_time_at_goal.add_(self.dt).mul_(goal_in_reach_mask.float())

        # Confirm completed tasks
        bot_done_mask = bot_time_at_goal >= T_TASK_CONFIRM
        bot_done_mask_f = bot_done_mask.float()
        bot_done_ctr = bot_done_ctr.add_(bot_done_mask_f)

        throughput = \
            bot_done_ctr.reshape(self.n_envs, -1).mean(-1) / torch.clip(env_run_times, 1.) * SCALE_THROUGHPUT

        # Check for terminal envs
        time_left = self.ep_duration - env_run_times
        rst_env_mask = time_left <= 0.

        # Expand env-wise data for each agent
        time_left = torch.repeat_interleave(time_left, self.n_bots, output_size=self.n_all_bots)
        bot_rst_mask = torch.repeat_interleave(rst_env_mask, self.n_bots, output_size=self.n_all_bots)

        throughput = torch.repeat_interleave(throughput, self.n_bots, output_size=self.n_all_bots)
        env_done_num = bot_done_mask_f.reshape(self.n_envs, -1).sum(-1)
        env_done_num = torch.repeat_interleave(env_done_num, self.n_bots, output_size=self.n_all_bots)

        # Evalute rewards
        # Score of 1 for own, score of 1/n_bots for shared rewards
        reward = bot_done_mask_f + env_done_num / self.n_bots

        # Assemble hidden state (should only be exposed to critics for better value estimation)
        goal_dir = torch.cat((goal_dir, self.zero_z), dim=-1)
        goal_path_dir = torch.cat((goal_path_dir, self.zero_z), dim=-1)

        loc_ori = bot_ori * self.quat_inv
        goal_dir = apply_quat_rot(loc_ori, goal_dir)
        goal_path_dir = apply_quat_rot(loc_ori, goal_path_dir)

        obs_aux = torch.stack([
            goal_dir[:, 0],
            goal_dir[:, 1],
            goal_path_dir[:, 0],
            goal_path_dir[:, 1],
            adjust_depth_range(goal_dist),
            adjust_depth_range(goal_path_len),
            goal_in_sight_mask.float(),
            bot_time_at_goal,
            bot_time_on_task * SCALE_TIME,
            time_left * SCALE_TIME,
            bot_done_mask_f,
            throughput], dim=1)

        return (
            env_run_times, bot_time_on_task, bot_time_at_goal,
            throughput, bot_done_ctr, bot_done_mask,
            bot_rst_mask, rst_env_mask, reward, obs_aux)

    def imu(self, ang_vel: Tensor, acc: Tensor, ori: Tensor) -> Tensor:
        """
        Simulate angular velocity, acceleration, and magnetic flux density,
        as if reported by the gyroscope, accelerometer, and magnetometer
        of a 9-DoF inertial measurement unit.
        """

        loc_ori = ori * self.quat_inv
        mag = apply_quat_rot(loc_ori, self.mag_ref)
        acc = apply_quat_rot(loc_ori, self.acc_ref + acc)

        acc = torch.tanh(acc / SCALE_ACC) * SCALE_ACC
        ang_vel = ang_vel * SCALE_ANG_VEL

        return torch.cat((ang_vel, acc, mag), axis=-1)

    @staticmethod
    def synaesthesia(rgb_sig: Tensor, pos: Tensor, rcvr_angles: Tensor) -> Tensor:
        """
        Weigh RGB signal transmissions of other agents by distance and sech filters
        wrt. 3 oriented receivers, each covering an angle of 120 degrees,
        to produce 3 RGB-like accumulations.

        While not quite realistic for robotic agents, this approach can be
        viewed as inferring information from 3 sound signals or pheromones
        in superposition.

        The RGB signals are also grounded by colouring the main bodies of agents
        with their emitted colour and their cargo with the colour of their target
        objective (hence union of senses).
        """

        # Pairwise differences
        pdiff_x = pos[:, :1] - pos[None, :, 0]
        pdiff_y = pos[:, 1:] - pos[None, :, 1]
        pdiff = torch.stack((pdiff_x, pdiff_y))

        # Weigh by distance, ignoring own signal
        pdist: Tensor = torch.linalg.norm(pdiff, dim=0)
        pdist.fill_diagonal_(MAX_DIST)
        pdist = adjust_depth_range(pdist)

        # NxNx3
        rgb_sig = torch.einsum('ij,ik->kij', rgb_sig, pdist)

        # Weigh by phase-shifted filters
        # Own angles are arbitrary, but pdist already masked the signals in mult
        incoming_angle = torch.atan2(pdiff_y, pdiff_x).T

        # NxNx1 - 1xNx3 -> NxNx3
        rel_angles = incoming_angle[..., None] - rcvr_angles[:, None]
        rel_angles = rel_angles + ((rel_angles < -torch.pi).float() - (rel_angles > torch.pi).float()) * _2PI

        # Cosh is naturally low at 120 degrees
        rel_weights = 1. / torch.cosh(rel_angles)

        # NxNx3, NxNx3 -> Nx3x3 -> Nx9
        rgb_agg = torch.einsum('ijk,ijl->ikl', rel_weights, rgb_sig).flatten(1)

        return rgb_agg

    def get_vector_observations(
        self,
        bot_pos: Tensor,
        bot_vel: Tensor,
        bot_old_vel: Tensor,
        bot_ori: Tensor,
        bot_ang_vel: Tensor,
        dof_vel: Tensor,
        act_trq: Tensor,
        act_rgb: Tensor,
        goal_rgb: Tensor,
        bot_time_at_goal: Tensor,
        throughput: Tensor
    ) -> Tensor:

        # NOTE: Differencing reported DOF pos. would be less noisy
        # https://forums.developer.nvidia.com/t/inaccuracy-of-dof-state-readings/197373/5
        dof_vel = dof_vel * SCALE_DOF

        bot_ang_vel = bot_ang_vel.reshape(-1, 3)
        bot_vel = bot_vel.reshape(-1, 3).contiguous()

        # IMU
        acc = (bot_vel - bot_old_vel) / self.dt
        obs_imu = self.imu(bot_ang_vel, acc, bot_ori)

        # Env local rgb/pos/ori
        rgb_loc = act_rgb.reshape(self.n_envs, -1, 3)
        pos_loc = bot_pos.reshape(self.n_envs, -1, 2)

        rcvr_ang = get_eulz_from_quat(bot_ori) + self.rcvr_rel_phases
        rcvr_ang_loc = rcvr_ang.reshape(self.n_envs, -1, 3)

        obs_rgb = torch.cat([self.synaesthesia(*rgb_pos_ori) for rgb_pos_ori in zip(rgb_loc, pos_loc, rcvr_ang_loc)])
        obs_rgb = torch.tanh(obs_rgb / SCALE_RCVR) * SCALE_RCVR

        # Assemble
        obs_vec = torch.cat((
            bot_time_at_goal[:, None], throughput[:, None],
            act_trq, dof_vel, obs_imu, act_rgb, goal_rgb, obs_rgb), dim=-1)

        bot_old_vel.copy_(bot_vel)

        return obs_vec

    @staticmethod
    def get_image_observations(images_rgb: 'list[Tensor]', images_depth: 'list[Tensor]') -> Tensor:
        rgb = torch.stack(images_rgb)[..., :3]
        dep = torch.stack(images_depth)

        # Normalise and stack channels
        rgb = rgb / 255.
        dep = adjust_depth_range(-dep)

        obs_img = torch.cat((rgb, dep[..., None]), dim=-1).permute(0, 3, 1, 2)

        return obs_img
