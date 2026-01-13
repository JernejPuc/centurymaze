"""ROS2 Recording Robot Node"""

import os
import traceback
from argparse import ArgumentParser
from time import perf_counter

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu
from mocap4r2_msgs.msg import RigidBodies


# ------------------------------------------------------------------------------
# MARK: RecBotNode

class RecBotNode(Node):
    DATA_DIR = os.path.expanduser('~/ros2_ws/records')

    NAME_RBID_MAP = {'camina': 42, 'james': 43}
    RBID_PIDX_MAP = {v: i for i, v in enumerate(NAME_RBID_MAP.values())}
    N_BOTS = len(NAME_RBID_MAP)

    IMG_RAW_WIDTH = 640
    IMG_RAW_HEIGHT = 480
    IMG_REC_DIMS = (320, 240)

    def __init__(self, name: str, n_to_record: int, recording_frequency: float):
        super().__init__('recbot_node')

        self.name = name
        self.rbody_id = self.NAME_RBID_MAP[self.name]
        self.n_to_record = n_to_record

        self.rgb_subscription = self.create_subscription(
            Image,
            f'/{self.name}/depth_camera/image_raw',
            self.rgb_callback,
            1)

        self.depth_subscription = self.create_subscription(
            Image,
            f'/{self.name}/depth_camera/depth/image_raw',
            self.depth_callback,
            1)

        self.imu_subscription = self.create_subscription(
            Imu,
            f'/{self.name}/imu',
            self.imu_callback,
            1)

        self.pose_subscription = self.create_subscription(
            RigidBodies,
            '/rigid_bodies',
            self.pose_callback,
            1)

        self.rec_timer = self.create_timer(
            1. / recording_frequency,
            self.rec_callback)

        self.callback_ctrs = np.zeros((5,), dtype=np.uint64)

        self.new_data: 'dict[str, np.ndarray | None]' = {k: None for k in ('rgb', 'depth', 'imu', 'poses')}
        self.rec_data: 'dict[str, np.ndarray]' = {
            'rgb': np.empty((n_to_record, *self.IMG_REC_DIMS[::-1], 3), dtype=np.uint8),
            'depth': np.empty((n_to_record, *self.IMG_REC_DIMS[::-1], 2), dtype=np.uint8),
            'imu': np.empty((n_to_record, 3 + 3 + 4), dtype=np.float32),
            'poses': np.empty((n_to_record, self.N_BOTS, 3 + 4)),
            'time': np.empty((n_to_record,)),
            'ctrs': np.empty((n_to_record, len(self.callback_ctrs)), dtype=np.uint64)}

        free_rec_idx = 1 + max((
            int(filename.split('_')[-1].split('.')[0])
            for filename in os.listdir(self.DATA_DIR) if filename.startswith('rec')), default=-1)

        self.rec_filename = os.path.join(self.DATA_DIR, f'rec_{free_rec_idx:02d}.npz')
        self.time_at_init = perf_counter()

    # --------------------------------------------------------------------------
    # MARK: sub_callbacks

    def rgb_callback(self, msg):
        self.new_data['rgb'] = \
            np.frombuffer(msg.data, dtype=np.uint8).reshape(self.IMG_RAW_HEIGHT, self.IMG_RAW_WIDTH, 3)

        self.callback_ctrs[0] += 1

    def depth_callback(self, msg):
        self.new_data['depth'] = \
            np.frombuffer(msg.data, dtype=np.uint8).reshape(self.IMG_RAW_HEIGHT, self.IMG_RAW_WIDTH, 2)

        self.callback_ctrs[1] += 1

    def imu_callback(self, msg):
        self.new_data['imu'] = np.array((
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w))

        self.callback_ctrs[2] += 1

    def pose_callback(self, msg):
        poses = np.zeros((self.N_BOTS, 7))

        for rigid_body in msg.rigidbodies:
            pose_idx = self.RBID_PIDX_MAP.get(int(rigid_body.rigid_body_name))

            if pose_idx is not None:
                poses[pose_idx] = (
                    rigid_body.pose.position.x,
                    rigid_body.pose.position.y,
                    rigid_body.pose.position.z,
                    rigid_body.pose.orientation.x,
                    rigid_body.pose.orientation.y,
                    rigid_body.pose.orientation.z,
                    rigid_body.pose.orientation.w)

        self.new_data['poses'] = poses
        self.callback_ctrs[3] += 1

    # --------------------------------------------------------------------------
    # MARK: rec_callback

    def rec_callback(self):
        if any(v is None for v in self.new_data.values()):
            return

        rec_step = self.callback_ctrs[-1]

        self.rec_data['rgb'][rec_step] = \
            cv2.resize(self.new_data['rgb'], self.IMG_REC_DIMS, interpolation=cv2.INTER_AREA)

        self.rec_data['depth'][rec_step] = \
            cv2.resize(self.new_data['depth'], self.IMG_REC_DIMS, interpolation=cv2.INTER_NEAREST_EXACT)

        self.rec_data['imu'][rec_step] = self.new_data['imu']
        self.rec_data['poses'][rec_step] = self.new_data['poses']
        self.rec_data['time'][rec_step] = perf_counter() - self.time_at_init
        self.rec_data['ctrs'][rec_step] = self.callback_ctrs

        self.callback_ctrs[-1] += 1
        self._logger.info('RGB: %d | DEPTH: %d | IMU: %d | POSES: %d | STEP: %d' % tuple(self.callback_ctrs))

        if self.callback_ctrs[-1] == self.n_to_record:
            np.savez_compressed(self.rec_filename, **self.rec_data)

            self._logger.info(f'Saved recording to: {self.rec_filename}')
            rclpy.shutdown()


# ------------------------------------------------------------------------------
# MARK: main

def main(args=None):
    parser = ArgumentParser()
    parser.add_argument('--name', type=str, default=None, help='Name of this robot.')
    parser.add_argument('--len', type=int, default=0, help='Num. of samples to record into a .npz file.')
    parser.add_argument('--freq', type=float, default=1., help='Recording frequency in samples per second.')
    parsed = parser.parse_args(args)

    if parsed.name is None:
        raise ValueError(f'`name` is a mandatory parameter (one of [{", ".join(RecBotNode.NAME_RBID_MAP)}]).')

    if parsed.len <= 0:
        raise ValueError('`len` is a mandatory parameter (> 0).')

    rclpy.init(args=args)
    node = RecBotNode(parsed.name, parsed.len, parsed.freq)
    node._logger.info('Node initialised. Running...')

    try:
        rclpy.spin(node)
        print('Done.')

    except KeyboardInterrupt:
        print('\nInterrupted.')

    except Exception:
        print(f'\n{traceback.format_exc()}\nException raised.')

    if 0 < (rec_step := node.callback_ctrs[-1]) < parsed.len:
        np.savez_compressed(node.rec_filename, **{k: v[:rec_step] for k, v in node.rec_data.items()})

        print(f'Saved partial recording to: {node.rec_filename}')

    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
