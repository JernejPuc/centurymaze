"""Project-wide configuration"""

import os


# Base parameters for procedural environment generation
# At higher levels, mazes become larger and more complex,
# with more agents and objectives and longer episodes
LEVEL_PARAMS = {
    1: {
        'env_width': 3.6,
        'n_grid_segments': 2,
        'n_graph_points': 5,
        'n_bots': 2,
        'n_objects': 2,
        'ep_duration': 15,
        'rng_seed': 12},
    2: {
        'env_width': 5.4,
        'n_grid_segments': 3,
        'n_graph_points': 9,
        'n_bots': 4,
        'n_objects': 3,
        'ep_duration': 30,
        'rng_seed': 42},
    3: {
        'env_width': 9.0,
        'n_grid_segments': 5,
        'n_graph_points': 16,
        'n_bots': 8,
        'n_objects': 4,
        'ep_duration': 60,
        'rng_seed': 5927},
    4: {
        'env_width': 12.6,
        'n_grid_segments': 7,
        'n_graph_points': 28,
        'n_bots': 16,
        'n_objects': 5,
        'ep_duration': 90,
        'rng_seed': 12},
    5: {
        'env_width': 18.0,
        'n_grid_segments': 10,
        'n_graph_points': 51,
        'n_bots': 32,
        'n_objects': 6,
        'ep_duration': 150,
        'rng_seed': 50},
    6: {
        'env_width': 25.2,
        'n_grid_segments': 14,
        'n_graph_points': 95,
        'n_bots': 64,
        'n_objects': 7,
        'ep_duration': 240,
        'rng_seed': 1},
    7: {
        'env_width': 36.0,
        'n_grid_segments': 20,
        'n_graph_points': 181,
        'n_bots': 128,
        'n_objects': 8,
        'ep_duration': 360,
        'rng_seed': 64578}}


# Assets
ASSET_DIR = os.path.abspath(os.path.join(__file__, '..', 'assets'))

BOT_WIDTH = 0.3
BOT_RADIUS = BOT_WIDTH/2 * 2**0.5
BOT_HEIGHT = 0.264

OBJECT_RADIUS = 0.225
OBJECT_HEIGHT = 0.275

WALL_WIDTH = 0.05
WALL_LENGTH = 1.8
WALL_HEIGHT = 0.75
WALL_HALFWIDTH = WALL_WIDTH / 2
WALL_HALFLENGTH = WALL_LENGTH / 2
WALL_HALFHEIGHT = WALL_HEIGHT / 2
ROOF_HEIGHT = WALL_HEIGHT - WALL_HALFWIDTH

# Some offset is added to prevent accidental collisions on startup
MIN_BUFFER = 0.025
WALL_HIDDEN_DEPTH = -(WALL_HALFHEIGHT + MIN_BUFFER)
ROOF_HIDDEN_DEPTH = -(ROOF_HEIGHT + MIN_BUFFER)

OBJECT_TO_WALL_BUFFER = OBJECT_RADIUS + WALL_HALFWIDTH + MIN_BUFFER
BOT_TO_BOT_BUFFER = BOT_RADIUS*2 + MIN_BUFFER
BOT_TO_OBJECT_BUFFER = OBJECT_RADIUS + BOT_RADIUS + MIN_BUFFER
BOT_TO_WALL_BUFFER = BOT_RADIUS + WALL_HALFWIDTH + MIN_BUFFER

MIN_GOAL_DIST = OBJECT_RADIUS + BOT_WIDTH/2
GOAL_RADIUS = 2*BOT_WIDTH + MIN_GOAL_DIST


# Colour palette
COLOURS = {
    'sky': (
        (230, 245, 255),),
    'pastel': (
        (185, 255, 175),
        (255, 255, 160),
        (255, 160, 205),
        (255, 205, 135),
        (175, 195, 255),
        (225, 160, 255),
        (175, 245, 255),
        (125, 235, 205),
        (255, 215, 215)),
    'basic': (
        (155, 205, 0),
        (0, 145, 0),
        (255, 145, 0),
        (0, 0, 255),
        (0, 160, 200),
        (255, 0, 0),
        (100, 50, 0),
        (115, 20, 165),
        (215, 20, 135)),
    'grey': (
        (0, 0, 0),
        (102, 102, 102),
        (127, 127, 127),
        (255, 255, 255))}

COLOURS = {clr_group: [[val / 255. for val in clr] for clr in clrs] for clr_group, clrs in COLOURS.items()}

# 9 + 2 colours: basic (objectives) + black & white (quiet or ungrounded)
RCVR_CLR_CLASSES = COLOURS['grey'][::3] + COLOURS['basic']

# 11 + 9 + 3 colours: above (com.) + pastel (walls) + other (bot, floor, sky)
ALL_CLR_CLASSES = RCVR_CLR_CLASSES + COLOURS['pastel'] + COLOURS['grey'][1:3] + COLOURS['sky']

N_RCVR_CLR_CLASSES = len(RCVR_CLR_CLASSES)
N_ALL_CLR_CLASSES = len(ALL_CLR_CLASSES)
N_OBJ_COLOURS = len(COLOURS['basic'])
N_WALL_COLOURS = len(COLOURS['pastel'])


# Segmentation classes for visual encoder pretraining
SEG_CLS_NULL = 0
SEG_CLS_PLANE = 1
SEG_CLS_WALL = 2
SEG_CLS_OBJ = 3
SEG_CLS_BOT = 4
SEG_CLS_BODY = 5
SEG_CLS_CARGO = 6
N_SEG_CLASSES = 7

# Model IO components
N_DOF_MOT = 4
N_DIM_RGB = 3
N_DIM_POS = 3

# 94 total:
# 4 torque cmd., 3x3 IMU, 4 ori. quat., 2 bot xy pos., 2 global frame bot xy vel.,
# 9 goal spec., 1 speaker spec., 1 time at goal, 1 own throughput,
# 11 colour cmd., 10x4 clr. channels, 10 clr. channel norm
OBS_LOC_SIZE = N_DOF_MOT + 3*3 + 4 + 2*(N_DIM_POS-1) + N_OBJ_COLOURS + 3 + N_RCVR_CLR_CLASSES
OBS_VEC_SIZE = OBS_LOC_SIZE + (N_RCVR_CLR_CLASSES-1) * (4+1)
OBS_ROLE_SLICE = slice(30, 31)

# Resolution mostly important for effective viewing distance
OBS_IMG_RES_WIDTH = 96
OBS_IMG_RES_HEIGHT = 48
OBS_IMG_CHANNELS = N_DIM_RGB + 1                                # RGB or HSV, depth (4)
ENC_IMG_CHANNEL_SPLIT = (OBS_IMG_CHANNELS, 2, 1)                # HSVD, clr. seg. + func. (type) seg., px. weights (7)
DEC_IMG_CHANNEL_SPLIT = (N_ALL_CLR_CLASSES, N_SEG_CLASSES, 1)   # Clr. softmax, func. softmax, depth value (31)
ALL_IMG_CHANNEL_SPLIT = ENC_IMG_CHANNEL_SPLIT + DEC_IMG_CHANNEL_SPLIT

# Torque cmd., rgb radial emitter
ACT_SPLIT = (N_DOF_MOT, N_DIM_RGB)
ACT_SIZE = sum(ACT_SPLIT)

ACT_DOF_MODES_BASE = [
    [0., 0., 0., 0.],
    [1., 1., 1., 1.],
    [-1., -1., -1., -1.],
    [-1., 1., -1., 1.],
    [1., -1., 1., -1.]]

# 28 total:
# 1 goal in frame, 9 obj. in frame, 2*9 obj. xy pos.
AUX_VAL_SPLIT = (1 + N_OBJ_COLOURS, 2*N_OBJ_COLOURS)
AUX_VAL_SIZE = sum(AUX_VAL_SPLIT)
AUX_VAL_SLICE = slice(OBS_VEC_SIZE + AUX_VAL_SPLIT[0], OBS_VEC_SIZE + AUX_VAL_SIZE)

# 1 heuristic clr. index
META_VAL_SIZE = 1

# 16 total:
# 1 avg. closest bot dist., 1 avg. vel. norm, 1 avg. goal delta, 1 avg. goal path len.,
# 1 avg. time on task, 1 avg. throughput, 9 throughput per obj., 1 time until end of episode
ENV_STAT_SIZE = 6 + N_OBJ_COLOURS + 1

# 112 total:
# 44 obs., 28 aux. val., 9 obj. prox., 3 air xy dir. & prox., 3 a* path xy dir. & prox.,
# 2 goal xy pos., # 1 own goal delta, 1 task completed, 1 time spent on task,
# 1 cell found, 1 bot ahead, 1 near bot prox. sum, 1 colliding flag,
# 16 env. stats.
STATE_VEC_SIZE = OBS_LOC_SIZE + N_OBJ_COLOURS + AUX_VAL_SIZE + 2*N_DIM_POS + (N_DIM_POS-1) + 7 + ENV_STAT_SIZE
STATE_VEC_SLICE = slice(OBS_VEC_SIZE, -META_VAL_SIZE)

# 16 total:
# 4 NESW cell connections, 2 wall hue & saturation, 9 obj. presence, 1 cell exploration
STATE_SPA_CHANNELS = 4 + 2 + N_OBJ_COLOURS + 1


# Training setup
DATA_DIR = 'data'   # Training meta data and model checkpoints
LOG_DIR = 'runs'    # Tracked scalars

# TODO: Override hardcoded `--n_envs 8` arg. in `runner`
N_ENVS = 8

# TODO: Override with `act_freq` arg. in `session`
STEPS_PER_SECOND = 4

# GAE length etc.
N_PASSES_PER_BATCH = 10
N_TRUNCATED_STEPS = 15  # STEPS_PER_SECOND * 4
N_ROLLOUT_STEPS = 120
N_ROLLOUTS_PER_EPOCH = 6
N_AUX_ITERS_PER_EPOCH = 8
SECONDS_PER_EPOCH = N_ROLLOUT_STEPS * N_ROLLOUTS_PER_EPOCH // STEPS_PER_SECOND

# Main + aux. updates
N_UPDATES_PER_EPOCH = (
    N_ROLLOUT_STEPS // N_TRUNCATED_STEPS * N_ROLLOUTS_PER_EPOCH * N_PASSES_PER_BATCH
    + N_ROLLOUT_STEPS * N_ROLLOUTS_PER_EPOCH // N_TRUNCATED_STEPS * N_AUX_ITERS_PER_EPOCH)

# 1 plot point and checkpoint per 3 virtual minutes, 20 per hour, 480 per day, 3360 per week
LOG_EPOCH_INTERVAL = max(1, 3 * 60 // SECONDS_PER_EPOCH)
CKPT_EPOCH_INTERVAL = LOG_EPOCH_INTERVAL

# 1 branch per half virtual hour, 48 per day, 336 per week
BRANCH_EPOCH_INTERVAL = 30 * 60 // SECONDS_PER_EPOCH

# 4-30 virtual minutes to warm up, 1-5 hours to train per stage, 10-24 hours total per agent
TIME_MILESTONE_MAP = {
    '128e-2a-15s': (4 * 60, 40 * 60, 40 * 60),
    '64e-4a-30s': (4 * 60, 80 * 60, 80 * 60),
    '32e-8a-60s': (4 * 60, 120 * 60, 120 * 60),
    '16e-16a-90s': (8 * 60, 160 * 60, 160 * 60),
    '8e-32a-150s': (12 * 60, 200 * 60, 200 * 60),
    '4e-64a-240s': (16 * 60, 240 * 60, 240 * 60),
    '2e-128a-360s': (28 * 60, 280 * 60, 320 * 60),
    '8e-32a-1m': (4 * 60, 120 * 60, 120 * 60),
    '8e-32c-1m': (4 * 60, 20 * 60, 20 * 60),
    '8e-32a-2m': (8 * 60, 120 * 60, 120 * 60),
    '8e-32a-3m': (12 * 60, 120 * 60, 120 * 60),
    '8e-64a-4m': (16 * 60, 120 * 60, 120 * 60),
    '8e-128a-5m': (20 * 60, 20 * 60, 120 * 60)}

N_EPOCHS_MAP = {k: tv[-1] // SECONDS_PER_EPOCH for k, tv in TIME_MILESTONE_MAP.items()}

UPDATE_MILESTONE_MAP = {
    k: tuple([v * N_UPDATES_PER_EPOCH // SECONDS_PER_EPOCH for v in tv])
    for k, tv in TIME_MILESTONE_MAP.items()}

WEIGHT_DECAY = 1e-4
VALUE_WEIGHT = 1.
AUX_WEIGHT = 1e-2
ENT_WEIGHT_MILESTONES = (1e-2, 1e-3)
