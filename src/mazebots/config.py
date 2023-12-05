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
        'ep_duration': 420,
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
DOF_VEC_SIZE = 4
XYZ_VEC_SIZE = 3
IMU_VEC_SIZE = 3 * XYZ_VEC_SIZE
RGB_VEC_SIZE = 3
RCVR_VEC_SIZE = 4

# 27 total:
# 1 confirmation status (time) at goal,
# 4 torque cmd., 4 whl. ang. vel., 3 vel., 3x3 IMU, 3 RGB cmd., 3 RGB task
OBS_VEC_SIZE = 1 + 2*DOF_VEC_SIZE + XYZ_VEC_SIZE + IMU_VEC_SIZE + 2*RGB_VEC_SIZE
OBS_RGB_SLICE = slice(OBS_VEC_SIZE-RGB_VEC_SIZE, OBS_VEC_SIZE)

# 50 total:
# 10x4 clr. channels, 10 clr. channel norm
OBS_COM_SIZE = (N_RCVR_CLR_CLASSES - 1) * (RCVR_VEC_SIZE + 1)

# Resolution mostly important for effective viewing distance
OBS_IMG_RES_WIDTH = 96
OBS_IMG_RES_HEIGHT = 48
OBS_IMG_CHANNELS = RGB_VEC_SIZE + 1                             # RGB or HSV, depth (4)
ENC_IMG_CHANNEL_SPLIT = (OBS_IMG_CHANNELS, 2, 1)                # HSVD, clr. seg. + func. (type) seg., px. weights (7)
DEC_IMG_CHANNEL_SPLIT = (N_ALL_CLR_CLASSES, N_SEG_CLASSES, 1)   # Clr. softmax, func. softmax, depth value (31)
ALL_IMG_CHANNEL_SPLIT = ENC_IMG_CHANNEL_SPLIT + DEC_IMG_CHANNEL_SPLIT

# Torque cmd., rgb radial emitter
ACT_VEC_SPLIT = (DOF_VEC_SIZE, RGB_VEC_SIZE)
ACT_VEC_SIZE = sum(ACT_VEC_SPLIT)

ACT_DOF_MODES_BASE = [
    [0., 0., 0., 0.],
    [1., 1., 1., 1.],
    [-1., -1., -1., -1.],
    [-1., 1., -1., 1.],
    [1., -1., 1., -1.]]

# 13 total:
# 9 prox. of obj. in fov., 1 goal in fov. mask, 2 air xy dir., 1 air prox.
AUX_VEC_SPLIT = (N_OBJ_COLOURS+1, XYZ_VEC_SIZE)
AUX_VEC_SIZE = sum(AUX_VEC_SPLIT)

GUIDE_VEC_SLICE = slice(OBS_VEC_SIZE + AUX_VEC_SPLIT[0], OBS_VEC_SIZE + AUX_VEC_SIZE)
OBS_EXT_SLICE = slice(None, OBS_VEC_SIZE + AUX_VEC_SIZE)

# 3 total:
# 1 nearest obj. proximity, 1 nearest obj. index, 1 last selected clr. index
META_VEC_SIZE = 3
META_VEC_SLICE = slice(-META_VEC_SIZE, None)

# 60 total:
# 27 obs., 13 aux.,
# 2 a* path xy dir., 1 a* path prox., 2 goal xy pos., 2 bot xy pos.,
# 4 bot proximity, 1 colliding flag,
# 1 time spent on task, 1 time until end of episode, 1 num. of completed tasks, 1 throughput, 1 dist. diff.,
# 1 nrg. in use, 1 nrg. used, 1 heat
STATE_VEC_SIZE = OBS_VEC_SIZE + AUX_VEC_SIZE + XYZ_VEC_SIZE + 2*(XYZ_VEC_SIZE-1) + RCVR_VEC_SIZE + 9
STATE_VEC_SLICE = slice(None, -META_VEC_SIZE)

# 43 total:
# 27 obs., 10 obj. ext., 3 guide ext., 3 meta
IPT_VEC_SPLIT = (OBS_VEC_SIZE, *AUX_VEC_SPLIT, META_VEC_SIZE)


# Training setup
DATA_DIR = 'data'   # Training meta data and model checkpoints
LOG_DIR = 'runs'    # Tracked scalars

# Should be a divisor of num. of all bots in the sim.
MINIBATCH_SIZE = 256
N_AUX_MINIBATCHES = 1

# GAE length etc.
STEPS_PER_SECOND = 4
N_TRUNCATED_STEPS = STEPS_PER_SECOND * 4
N_ROLLOUT_STEPS = 16#240
N_ROLLOUTS_PER_EPOCH = 6
N_AUX_ITERS_PER_EPOCH = 8
SECONDS_PER_EPOCH = N_ROLLOUT_STEPS * N_ROLLOUTS_PER_EPOCH // STEPS_PER_SECOND

# Main + aux. updates
N_UPDATES_PER_EPOCH = (
    N_ROLLOUT_STEPS // N_TRUNCATED_STEPS * N_ROLLOUTS_PER_EPOCH
    + N_ROLLOUT_STEPS * N_ROLLOUTS_PER_EPOCH // N_TRUNCATED_STEPS * N_AUX_ITERS_PER_EPOCH)

# 1 plot point and checkpoint per 4 virtual minutes, 15 per hour, 360 per day, 2520 per week
LOG_EPOCH_INTERVAL = max(1, 4 * 60 // SECONDS_PER_EPOCH)
CKPT_EPOCH_INTERVAL = LOG_EPOCH_INTERVAL

# 1 branch per half virtual hour, 48 per day, 360 per week
BRANCH_EPOCH_INTERVAL = 30 * 60 // SECONDS_PER_EPOCH

# 4-30 virtual minutes to warm up, 14+5 hours to train, 5 hours to cool down, 24 hours total
TIME_MILESTONE_MAP = {
    1: (4 * 60, 40 * 60, 40 * 60),
    2: (4 * 60, 80 * 60, 80 * 60),
    3: (4 * 60, 120 * 60, 120 * 60),
    4: (8 * 60, 160 * 60, 160 * 60),
    5: (12 * 60, 200 * 60, 200 * 60),
    6: (16 * 60, 240 * 60, 240 * 60),
    7: (28 * 60, 280 * 60, 320 * 60)}

N_EPOCHS_MAP = {k: tv[-1] // SECONDS_PER_EPOCH for k, tv in TIME_MILESTONE_MAP.items()}

UPDATE_MILESTONE_MAP = {
    k: tuple([v * N_UPDATES_PER_EPOCH // SECONDS_PER_EPOCH for v in tv])
    for k, tv in TIME_MILESTONE_MAP.items()}

WEIGHT_DECAY = 1e-2
VALUE_WEIGHT = 1.
AUX_WEIGHT = 1e-2
ENT_WEIGHT_MILESTONES = (1e-2, 1e-3)
