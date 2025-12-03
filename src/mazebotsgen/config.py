"""Project-wide configuration"""

import os


# Base parameters for procedural generation:
# At higher levels, mazes become larger and more complex, with more agents and longer episodes
LEVEL_PARAMS = {
    1: {
        'side_length': 10.8,
        'n_graph_points': 20,
        'n_bots': 6,
        'n_goals': 2,
        'ep_duration': 27},
    2: {
        'side_length': 14.4,
        'n_graph_points': 30,
        'n_bots': 12,
        'n_goals': 4,
        'ep_duration': 36},
    3: {
        'side_length': 18.0,
        'n_graph_points': 50,
        'n_bots': 24,
        'n_goals': 6,
        'ep_duration': 45},
    4: {
        'side_length': 25.2,
        'n_graph_points': 90,
        'n_bots': 48,
        'n_goals': 8,
        'ep_duration': 60},
    5: {
        'side_length': 36.0,
        'n_graph_points': 180,
        'n_bots': 96,
        'n_goals': 8,
        'ep_duration': 90}}


# ------------------------------------------------------------------------------
# MARK: Assets

ASSET_DIR = os.path.abspath(os.path.join(__file__, '..', 'assets'))

BOT_BODY_IDX = 0
BOT_LOAD_IDX = 1

SIDE_W_IDX = 0
SIDE_N_IDX = 1

BOT_WIDTH = 0.3
BOT_RADIUS = BOT_WIDTH/2 * 2**0.5
BOT_HEIGHT = 0.264

# Lowered inside the body, otherwise the payload was clipping in all tested configurations
CAM_OFFSET = (BOT_WIDTH/2 - 0.025, 0., BOT_HEIGHT - 0.1)

OBJ_RADIUS = 0.319
OBJ_HEIGHT = 0.225

WALL_WIDTH = 0.05
WALL_LENGTH = 1.8
WALL_HEIGHT = WALL_LENGTH
CELL_DIAG_LENGTH = WALL_LENGTH * 2**0.5
N_CELL_DIVS = 2

BLOCK_HEIGHT = WALL_HEIGHT
BLOCK_HALFHEIGHT = BLOCK_HEIGHT/2
BLOCK_HIDDENHEIGHT = -(BLOCK_HALFHEIGHT + WALL_WIDTH)

# Some offset is added to prevent accidental collisions on startup
MIN_SPAWN_GAP = 0.025

OBJ_TO_WALL_GAP = OBJ_RADIUS + WALL_WIDTH/2 + MIN_SPAWN_GAP
BOT_TO_WALL_GAP = BOT_RADIUS + WALL_WIDTH/2 + MIN_SPAWN_GAP
OBJ_TO_OBJ_GAP = OBJ_RADIUS*2 + MIN_SPAWN_GAP
BOT_TO_OBJ_GAP = BOT_RADIUS + OBJ_RADIUS + MIN_SPAWN_GAP
BOT_TO_BOT_GAP = BOT_RADIUS*2 + MIN_SPAWN_GAP

# Objects are biased to spawn slightly to the side instead of directly at the middle of a corridor
MAX_OBJ_TO_WALL_DIST = WALL_LENGTH / 3


# ------------------------------------------------------------------------------
# MARK: Sim params.

MOT_STIFFNESS = 0.001   # Bot won't move at 0.1, slow turning difficult at 0.01
MOT_DAMPING = 0.02      # Applying torque without damping makes the bot fly off
WHL_FRICTION = 0.01
MOT_MAX_TORQUE = 1.     # Per motor/wheel
GRAV_CONST = 9.80665
BASE_LIGHT_INTENSITY = 0.2
BASE_LIGHT_AMBIENT = 0.8
LIGHT_RAND_PROB = 0.5

MAX_PARAM_OFFSET = 0.05
MAX_DURATION_OFFSET = 0.1
IO_NOISE_SCALE = 0.03
MAX_INTENSITY_OFFSET = 0.1
MAX_AMBIENT_OFFSET = 0.2


# ------------------------------------------------------------------------------
# MARK: Task params.

MIN_GOAL_DIST = OBJ_RADIUS + BOT_WIDTH/2
MAX_GOAL_DIST = MIN_GOAL_DIST + 50.     # (ENV_DIAG_LENGTH - (WALL_WIDTH + BOT_RADIUS + OBJ_RADIUS)) * 2**0.5
GOAL_PRED_RADIUS = 4.                   # ((ENV_SIDE_LENGTH / 5)**2 / pi)**0.5
GOAL_ZONE_RADIUS = 1.                   # MIN_GOAL_DIST + BOT_WIDTH|RADIUS*2
PROX_SIGNAL_RADIUS = 1.                 # 2.5 * BOT_RADIUS

MAX_COORD_VAL = 18.                     # ENV_SIDE_LENGTH / 2
MAX_IMG_DEPTH = 24.                     # Longest unobstructed line of sight
ENV_HALFSPACING = 0.1

STEPS_PER_SECOND = 4
MAX_EP_DURATION = 90
VIS_EP_DURATION = 5

# Joint, inf. horizon
OBJ_FOUND_RWD = 0.15        # So that 8 eventual rewards sum to 1.2
BLIF_DELTA_RWD = 0.15       # To be averaged over assigned bots per obj. until completion

# Individual, inf. horizon
CELL_REACHED_RWD = 0.02     # Signed wrt. path delta after goal is found by any, ideal remainder when goal is reached
MAX_GOAL_REACHED_RWD = 1.2  # EP_DURATION * STEPS_PER_SECOND * (MAX_PATH_DELTA / CELL_SIDE_LENGTH) * CELL_REACHED_RWD

# Individual, short horizon
PROXIMITY_RWD = -0.02       # Negative of cell reached rwd.
COLLISION_RWD = -0.01       # PROXIMITY_RWD / 2

# Discount factors wrt. different reward categories
DISCOUNTS = (
    1.,                                     # Joint rewards
    1.,                                     # Long-horizon rewards: no half-life
    0.5 ** (1. / (3 * STEPS_PER_SECOND)))   # Short-horizon rewards: half-life at 3 seconds, gamma 0.944


# ------------------------------------------------------------------------------
# MARK: Colour palette

# 19 total: 3 background (sky, floor/clutter, bot body) + 8 wall + 8 goal/payload
# TODO: Use grey-127 for separate clutter colour class
COLOURS = {
    'neutral': (
        (230, 245, 255),    # Pale cyan
        (168, 168, 168),    # Grey
        (255, 255, 255)),   # White
    'wall': (
        (185, 255, 175),    # Light green
        (255, 160, 205),    # Pink
        (175, 245, 255),    # Light cyan
        (225, 225, 225),    # Light grey
        (225, 160, 255),    # Violet
        (255, 255, 160),    # Yellow
        (255, 205, 135),    # Light orange
        (125, 235, 205)),   # Turquoise
    'goal': (
        (155, 205, 0),      # Yellow green
        (0, 145, 0),        # Green
        (240, 135, 0),      # Dark orange
        (0, 0, 255),        # Blue
        (0, 160, 200),      # Dark cyan
        (255, 0, 0),        # Red
        (255, 215, 0),      # Golden
        (160, 20, 150))}    # Dark magenta

COLOURS = {clr_group: [[val / 255. for val in clr] for clr in clrs] for clr_group, clrs in COLOURS.items()}

N_WALL_CLRS = len(COLOURS['wall'])
N_GOAL_CLRS = len(COLOURS['goal'])
N_CLR_CLASSES = sum(len(clrs) for clrs in COLOURS.values())

SKY_CLR_IDX = 0
FLOOR_CLR_IDX = 1
WHITE_CLR_IDX = 2

MIN_CLR_CLS_CONFIDENCE = 0.66


# ------------------------------------------------------------------------------
# MARK: Visual entity classes

ENT_CLS_SKY = 0
ENT_CLS_FLOOR = 1
ENT_CLS_WALL = 2
ENT_CLS_CLUTTER = 3
ENT_CLS_OBJ = 4
ENT_CLS_BOT = 5
ENT_CLS_PAYLOAD = 6
N_ENT_CLASSES = 7

WALL_CLS_OFFSET = N_ENT_CLASSES
OBJ_CLS_OFFSET = WALL_CLS_OFFSET + N_WALL_CLRS
BOT_CLS_OFFSET = OBJ_CLS_OFFSET + N_GOAL_CLRS
PLD_CLS_OFFSET = BOT_CLS_OFFSET + N_GOAL_CLRS
N_ALL_CLASSES = N_ENT_CLASSES + N_WALL_CLRS + N_GOAL_CLRS*2 + 1


# ------------------------------------------------------------------------------
# MARK: IO components

N_DOF_MOT = 4
N_DIM_RGB = 3
N_DIM_POS = 2
N_DIM_DIRLEN = 3
N_SENS_PROX = 4
N_CARDINALS = 4

# 28 total:
# 4 torque cmd., 1 broadcast cmd., 2 bot pos. xy, 2 global frame bot vel. xy,
# 2 accel. xy, 1 ang. vel. z, 2 sin-cos z, 4 prox. sensors,
# 8 goal spec., 2 optional/oracular goal pos. xy
OBS_VEC_SIZE = N_DOF_MOT + 1 + N_DIM_POS*3 + 3 + N_SENS_PROX + N_GOAL_CLRS + 2
COM_ON_IDX = N_DOF_MOT
BOT_POS_IDX = N_DOF_MOT + 1
BOT_POS_SLICE = slice(BOT_POS_IDX, BOT_POS_IDX + 2)
GOAL_SPEC_SLICE = slice(OBS_VEC_SIZE - N_GOAL_CLRS - 2, OBS_VEC_SIZE - 2)

# Resolution mostly important for effective viewing distance
OBS_IMG_RES_WIDTH = 96
OBS_IMG_RES_HEIGHT = 48
OBS_IMG_CHANNELS = N_DIM_RGB + 1                                # RGB or HSV, depth (4)
DEC_IMG_CHANNEL_SPLIT = (N_CLR_CLASSES, N_ENT_CLASSES, 1)       # Clr. softmax, ent. softmax, depth value (27)

# Torque cmd., com. cmd.
ACT_SPLIT = (N_DOF_MOT, 1)
ACT_SIZE = sum(ACT_SPLIT)

ACT_VALUES = (
    [0., 0., 0., 0., 0.],
    [1., 1., 1., 1., 0.],
    [-1., -1., -1., -1., 0.],
    [-1., 1., -1., 1., 0.],
    [1., -1., 1., -1., 0.],
    [0., 0., 0., 0., 1.],
    [1., 1., 1., 1., 1.],
    [-1., -1., -1., -1., 1.],
    [-1., 1., -1., 1., 1.],
    [1., -1., 1., -1., 1.])

BELIEF_SPLIT = (N_GOAL_CLRS, N_DIM_POS)

# 51 total:
# 1 redundant (leftover for consistency with proto. IO scheme),
# 1 avg. closest bot dist., 1 avg. vel. norm,
# 8 tasks left per obj., 8 avg. path len. per obj., 8 min. path len. per obj.,
# 8 obj. found, 16 obj. pos xy
STATE_ENV_SIZE = 3 + N_GOAL_CLRS*6
STATE_ENV_SLICE = slice(OBS_VEC_SIZE-1, OBS_VEC_SIZE-1 + STATE_ENV_SIZE)

# 30 total:
# 8 obj. dist., 8 obj. in frame,
# 2 goal pos. xy, 3 a* path dir. & len. to goal, 3 air dir. & dist. to goal,
# 1 goal found, 1 goal belief correct, 1 diff. of path to goal within cell,
# 1 cumulative per-cell rwd., 1 near ent. prox. sum, 1 colliding flag
STATE_BOT_SIZE = N_GOAL_CLRS*2 + N_DIM_POS + N_DIM_DIRLEN*2 + 6
GOAL_FOUND_IDX = OBS_VEC_SIZE + STATE_ENV_SIZE-1 + STATE_BOT_SIZE-6
GOAL_FOUND_SLICE = slice(GOAL_FOUND_IDX, GOAL_FOUND_IDX+1)

# 108 total:
# 28 obs., 50 common, 30 specific
STATE_VEC_SIZE = OBS_VEC_SIZE + STATE_ENV_SIZE-1 + STATE_BOT_SIZE

# 15 total:
# 8 obj. presence, 4 NWSE cell walls, 1 cell clr. seg., 1 bot presence, 1 cell exploration
STATE_LAYOUT_CHANNELS = N_GOAL_CLRS + N_CARDINALS + 1
STATE_MAP_CHANNELS = STATE_LAYOUT_CHANNELS + 2

# 11 total:
# 8 obj. in frame + 1 for one-hot, 2 goal pos. xy
AUX_VAL_SPLIT = (N_GOAL_CLRS+1, N_DIM_POS)
AUX_VAL_SIZE = N_GOAL_CLRS + N_DIM_POS
AUX_VAL_OFFSET = OBS_VEC_SIZE + STATE_ENV_SIZE-1 + N_GOAL_CLRS
AUX_VAL_SLICE = slice(AUX_VAL_OFFSET, AUX_VAL_OFFSET + AUX_VAL_SIZE)


# ------------------------------------------------------------------------------
# MARK: Networks

COM_ORACULAR = 2
COM_TARGET = 1
COM_NONE = 0

AUX_DETACH = 3
AUX_REPLAY = 2
AUX_ONLINE = 1
AUX_NONE = 0

RWD_ALL = 3
RWD_UTIL = 2
RWD_GAIN = 1
RWD_DEFAULT = 0

AGENT_TYPE_CONFIGS = {
    'base-nocom': {'com_mode': COM_NONE, 'aux_mode': AUX_NONE, 'rwd_mode': RWD_DEFAULT},
    'nobl-dial': {'com_mode': COM_TARGET, 'aux_mode': AUX_NONE, 'rwd_mode': RWD_UTIL},
    'dial-oracle': {'com_mode': COM_ORACULAR, 'aux_mode': AUX_NONE, 'rwd_mode': RWD_ALL},
    'diabl-infer': {'com_mode': COM_TARGET, 'aux_mode': AUX_REPLAY, 'rwd_mode': RWD_ALL}}

K_DIM = 32
V_DIM = 64
QKV_SPLIT = (K_DIM, K_DIM, V_DIM)

N_HEADS = 4
ATN_ENC_SIZE = N_HEADS * K_DIM

VIS_ENC_SIZE = 224
MAP_ENC_SIZE = 116
POL_ENC_SIZE = 320
VAL_ENC_SIZE = 512
PRE_OUT_SIZE = 256

MEM_SIZE = 256
CHRONO_RANGES = (1/STEPS_PER_SECOND, 1, 3, 10, 30)
CHRONO_SIZE = MEM_SIZE // (len(CHRONO_RANGES) - 1)


# ------------------------------------------------------------------------------
# MARK: Vis. enc.

PX_WEIGHT_MAP = {
    'SKY': 0.05,
    'FLOOR': 0.025,
    'WALL': 0.111,
    'CLUTTER': 0.65,
    'OBJ': 1.55,
    'BODY': 1.0,
    'PAYLOAD': 2.575}


# ------------------------------------------------------------------------------
# MARK: Training setup

DATA_DIR = 'data'   # Training meta data and model checkpoints
LOG_DIR = 'runs'    # Tracked scalars

ENV_NUM_LVL_PRESETS = {
    '1x1': {1: 1},
    '2x1': {1: 2},
    '1x2': {2: 1},
    '1x3': {3: 1},
    '1x4': {4: 1},
    '1x5': {5: 1},
    '44x1+4': {1: 44, 2: 1, 3: 1, 4: 1, 5: 1},
    '21x2+3': {2: 21, 3: 1, 4: 1, 5: 1},
    '10x3+2': {3: 10, 4: 1, 5: 1},
    '5x4+1': {4: 5, 5: 1},
    '3x5+0': {5: 3},
    'all': {1: 1, 2: 1, 3: 1, 4: 1, 5: 1},
    'prog': {1: 5, 2: 4, 3: 3, 4: 2, 5: 1}}

N_PROG_BOTS = sum(LEVEL_PARAMS[lvl]['n_bots'] * n_envs for lvl, n_envs in ENV_NUM_LVL_PRESETS['prog'].items())
VIS_BATCH_SIZE = N_PROG_BOTS // 3

# Runner args.
SEEDS = (42, 9, 7, 5, 3, 1)  # HHGTTG + RoP + 5 to complete odd number sequence

N_MIN_EVAL_EPS = 300
N_MIN_EVAL_STEPS = 100_000
LEVEL_EVAL_STEPS = {
    level: max(N_MIN_EVAL_STEPS, params['ep_duration'] * STEPS_PER_SECOND * N_MIN_EVAL_EPS)
    for level, params in LEVEL_PARAMS.items()}

# RL args.
BATCH_SIZE = N_PROG_BOTS
N_TRUNCATED_STEPS = 20  # 5 * STEPS_PER_SECOND
COM_BUFFER_SIZE = 4320  # 3 (N_EPS) * 9 (N_ENVS) * 8 (N_GOALS) * 20 (N_TRUNC)
BUFFER_SIZE = 1000      # 3 (N_EPS) * 90*4 (STEPS_PER_EP); last 4 min
N_ROLLOUT_STEPS = 64    # BUFFER_SIZE / 15 (N_SAMPLE_UPDATES); with slight offset
N_PASSES_PER_STEP = 1
SECONDS_PER_EPOCH = N_ROLLOUT_STEPS // STEPS_PER_SECOND

# 1 plot point per 16 virtual seconds, 225 per hour, 3600 in 16 hours
LOG_EPOCH_INTERVAL = 1
CKPT_EPOCH_INTERVAL = 12

# 1 branch per 80 minutes, 12 in 16 hours
BRANCH_EPOCH_INTERVAL = 80 * 60 // SECONDS_PER_EPOCH

# Vis. training is online and single-env.
VIS_LOG_INTERVAL = 160
VIS_CKPT_INTERVAL = 3 * VIS_LOG_INTERVAL
VIS_BRANCH_INTERVAL = 45 * VIS_LOG_INTERVAL

# Separate schedules btw. network modules
# 10 virtual minutes to warm up, 16 hours per agent to train and anneal, almost 8 months over all 342 agents
UPDATE_MAP = {
    'policy': {'milestones': (10 * 90, 3 * 3600, 16 * 3600), 'lr': 8e-5, 'div': (20., 400.)},
    'critic': {'milestones': (1 * 90, 3 * 3600, 15 * 3600), 'lr': 3e-4, 'div': (20., 6.)},
    'visenc': {'milestones': (2 * 60, 15 * 3600, 24 * 3600), 'lr': 6e-4, 'div': (20., 400.)}}

UPDATE_MILESTONE_MAP = {
    'policy': tuple([v // SECONDS_PER_EPOCH for v in UPDATE_MAP['policy']['milestones']]),
    'critic': tuple([v // SECONDS_PER_EPOCH for v in UPDATE_MAP['critic']['milestones']]),
    'visenc': tuple([v * STEPS_PER_SECOND for v in UPDATE_MAP['visenc']['milestones']])}

WEIGHT_DECAY = 1e-4
VALUE_WEIGHT = 1.
AUX_WEIGHT = 0.2
ENT_WEIGHT_MILESTONES = (5e-3, 1e-4)
TRACE_LAMBDA = 0.9
CLIP_RATIO = 0.125
