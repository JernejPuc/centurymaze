"""Project-wide configuration"""

import os


LEVEL_PARAMS = {
    1: {
        'side_length': 3.2,
        'wall_length': 0.8,
        'n_graph_points': 30,
        'n_bots': 2,
        'n_goals': 1,
        'ep_duration': 30},
    2: {
        'side_length': 25.6,
        'wall_length': 1.6,
        'n_graph_points': 100,
        'n_bots': 48,
        'n_goals': 8,
        'ep_duration': 90}}


# ------------------------------------------------------------------------------
# MARK: Assets

ASSET_DIR = os.path.abspath(os.path.join(__file__, '..', 'assets'))
ALT_ASSET_DIR = os.path.abspath(os.path.join(__file__, '..', '..', 'mazebotsgen', 'assets'))

SIDE_W_IDX = 0
SIDE_N_IDX = 1

BOT_BODY_IDX = 0
BOT_POLE_IDX = 1
BOT_CAM_IDX = 2
BOT_CHAS_IDX = 3
BOT_WHL1_IDX = 4
N_BOT_WHLS = 4

BOT_WIDTH = 0.425
BOT_RADIUS = BOT_WIDTH/2 * 2**0.5
BOT_HEIGHT = 0.235
CAM_HEIGHT = 0.405

CAM_OFFSET = (-0.1475, 0., CAM_HEIGHT)
CAM_TARGET_OFFSET = (BOT_WIDTH, 0., 0.)
FLW_OFFSET = (-BOT_WIDTH, 0., 0.)

OBJ_RADIUS = 0.1 * 2**0.5
OBJ_HEIGHT = 0.1

WALL_WIDTH = 0.05
WALL_HEIGHT = 0.75
MAX_WALL_LENGTH = 1.6
CELL_DIAG_LENGTH = MAX_WALL_LENGTH * 2**0.5
MAX_SIDE_HALFLENGTH = 25.6
MAX_SIDE_DIVS = 16

BLOCK_HEIGHT = WALL_HEIGHT
BLOCK_HALFHEIGHT = BLOCK_HEIGHT/2
BLOCK_HIDDENHEIGHT = -(BLOCK_HALFHEIGHT + WALL_WIDTH)

MIN_SPAWN_GAP = 0.025
OBJ_TO_WALL_GAP = OBJ_HEIGHT + WALL_WIDTH/2
BOT_TO_WALL_GAP = BOT_RADIUS + WALL_WIDTH/2 + MIN_SPAWN_GAP
OBJ_TO_OBJ_GAP = OBJ_RADIUS*2
BOT_TO_OBJ_GAP = BOT_RADIUS + OBJ_RADIUS + MIN_SPAWN_GAP
BOT_TO_BOT_GAP = BOT_RADIUS*2 + MIN_SPAWN_GAP


# ------------------------------------------------------------------------------
# MARK: Sim params.

MOT_STIFFNESS = 0.001
MOT_DAMPING = 0.02
WHL_FRICTION = 0.01
MOT_MAX_TORQUE = 1.
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
MAX_GOAL_DIST = MIN_GOAL_DIST + 50.
GOAL_PRED_RADIUS = 4.
GOAL_ZONE_RADIUS = 1.
PROX_SIGNAL_RADIUS = 1.

MAX_COORD_VAL = 12.6
MAX_IMG_DEPTH = 12.
DEPTH_BIN_STEP = 0.2
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
# MARK: Visual entity classes

ENT_CLS_PLANE = 0
ENT_CLS_WALL = 1
ENT_CLS_CLUTTER = 2
ENT_CLS_OBJ = 3
ENT_CLS_BOT = 4
N_ENT_CLASSES = 5

OBJ_CLS_OFFSET = N_ENT_CLASSES

# ------------------------------------------------------------------------------
# MARK: Textures & colour palette

TEXT_SPLIT = 5
TEXT_MIXED = 4
TEXT_RANDAUG = 3
TEXT_ROOM = 2
TEXT_WARES = 1
TEXT_NONE = 0

AUG_DECAL = 2
AUG_BASE = 1
AUG_NONE = 0

COLOURS = {
    'neutral': (
        (0, 0, 0),          # Black
        (100, 100, 100),    # Dark grey
        (150, 150, 150),    # Grey
        (200, 200, 200),    # Light grey
        (255, 255, 255)),   # White
    'wall': (
        (255, 205, 135),    # Light orange
        (255, 255, 160),    # Yellow
        (185, 255, 175),    # Light green
        (125, 235, 205),    # Turquoise
        (175, 195, 255),    # Pale blue
        (175, 245, 255),    # Light cyan
        (225, 160, 255),    # Violet
        (255, 160, 205)),   # Pink
    'goal': (
        (190, 0, 0),        # Red
        (240, 135, 0),      # Dark orange
        (255, 215, 0),      # Golden
        (155, 205, 0),      # Yellow green
        (0, 145, 0),        # Green
        (0, 160, 200),      # Dark cyan
        (0, 0, 255),        # Blue
        (150, 20, 170)),    # Purple
    'entity': (
        (0, 0, 0),          # Black (floor/sky)
        (0, 0, 255),        # Blue (wall)
        (255, 255, 255),    # White (clutter)
        (255, 0, 0),        # Red (goal object)
        (0, 255, 0))}       # Green (bot)

SKY_CLR_IDX = 0
WHEEL_CLR_IDX = 1
FLOOR_CLR_IDX = 2
BOT_CLR_IDX = 3
WHITE_CLR_IDX = 4

CLR_CLS_UNSPEC = 8  # N_GOAL_CLRS

CLR_CLS_CLRS = COLOURS['goal'] + (COLOURS['neutral'][WHITE_CLR_IDX],)
ENT_CLS_CLRS = COLOURS['entity']
ALL_CLS_CLRS = COLOURS['goal'] + COLOURS['wall'] + COLOURS['neutral']

COLOURS = {clr_group: [[val / 255. for val in clr] for clr in clrs] for clr_group, clrs in COLOURS.items()}

N_WALL_CLRS = len(COLOURS['wall'])
N_GOAL_CLRS = len(COLOURS['goal'])
N_CLR_CLASSES = len(CLR_CLS_CLRS)

MIN_CLR_CLS_CONFIDENCE = 0.66


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
OBS_IMG_CHANNELS = N_DIM_RGB + 1                                # HSV, depth (4)
DEC_IMG_CHANNEL_SPLIT = (N_CLR_CLASSES, N_ENT_CLASSES, 1)       # Clr. softmax, ent. softmax, depth value (15)

# Torque cmd., com. cmd.
ACT_SPLIT = (N_DOF_MOT, 1)
ACT_SIZE = sum(ACT_SPLIT)

ACT_VALUES = (
    [0., 0., 0., 0., 0.],
    [1., 1., 1., 1., 0.],
    [-1., -1., -1., -1., 0.],
    [-1.5, 1.5, -1.5, 1.5, 0.],
    [1.5, -1.5, 1.5, -1.5, 0.],
    [0., 0., 0., 0., 1.],
    [1., 1., 1., 1., 1.],
    [-1., -1., -1., -1., 1.],
    [-1.5, 1.5, -1.5, 1.5, 1.],
    [1.5, -1.5, 1.5, -1.5, 1.])

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
    ENT_CLS_PLANE: 0.075,
    ENT_CLS_WALL: 0.111,
    ENT_CLS_CLUTTER: 1.65,
    ENT_CLS_OBJ: 2.75,
    ENT_CLS_BOT: 1.}


# ------------------------------------------------------------------------------
# MARK: Training setup

DATA_DIR = 'data'   # Training meta data and model checkpoints
LOG_DIR = 'runs'    # Tracked scalars

ENV_NUM_LVL_PRESETS = {
    '1x1': {1: 1},
    '2x1': {1: 2},
    '1x2': {2: 1},
    '2x2': {2: 2},
    'all': {1: 1, 2: 1},
    'all2': {1: 2, 2: 2},
    'vis': {1: 8, 2: 2},
    'prog': {1: 20, 2: 5}}

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
N_TRUNCATED_STEPS = 20
COM_BUFFER_SIZE = 4320
BUFFER_SIZE = 1000
N_ROLLOUT_STEPS = 64
N_PASSES_PER_STEP = 1
SECONDS_PER_EPOCH = N_ROLLOUT_STEPS // STEPS_PER_SECOND

# 1 plot point per 16 virtual seconds, 225 per hour, 3600 in 16 hours
LOG_EPOCH_INTERVAL = 1
CKPT_EPOCH_INTERVAL = 12

# 1 branch per 80 minutes, 12 in 16 hours
BRANCH_EPOCH_INTERVAL = 80 * 60 // SECONDS_PER_EPOCH

# Vis. training is online and single-batch
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
