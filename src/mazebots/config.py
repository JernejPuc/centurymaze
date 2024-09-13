"""Project-wide configuration"""

import os


# ------------------------------------------------------------------------------
# MARK: Assets

ASSET_DIR = os.path.abspath(os.path.join(__file__, '..', 'assets'))
BOT_FILE_NAME = 'mazebot.urdf'
OBJ_FILE_NAME = 'levisphere.urdf'

OBJ_BODY_IDX = 0
BOT_BODY_IDX = 0
BOT_BEACON_IDX = 1

SIDE_W_IDX = 0
SIDE_N_IDX = 1

BOT_WIDTH = 0.3
BOT_RADIUS = BOT_WIDTH/2 * 2**0.5
BOT_HEIGHT = 0.264

MOT_STIFFNESS = 0.001   # Bot won't move at 0.1, slow turning difficult at 0.01
MOT_DAMPING = 0.02      # Applying torque without damping makes the bot fly off
MOT_MAX_TORQUE = 1.     # Per motor/wheel

# Lowered inide the body, otherwise the beacon was clipping in all tested configurations
CAM_OFFSET = (BOT_WIDTH/2 - 0.025, 0., BOT_HEIGHT - 0.1)

OBJ_RADIUS = 0.225
OBJ_HEIGHT = 0.275

WALL_WIDTH = 0.05
WALL_LENGTH = 1.8
WALL_HEIGHT = 0.75
CELL_DIAG_LENGTH = WALL_LENGTH * 2**0.5

# Some offset is added to prevent accidental collisions on startup
MIN_SPAWN_GAP = 0.025

OBJ_TO_WALL_GAP = OBJ_RADIUS + WALL_WIDTH/2 + MIN_SPAWN_GAP
BOT_TO_WALL_GAP = BOT_RADIUS + WALL_WIDTH/2 + MIN_SPAWN_GAP
BOT_TO_OBJ_GAP = BOT_RADIUS + OBJ_RADIUS + MIN_SPAWN_GAP
BOT_TO_BOT_GAP = BOT_RADIUS*2 + MIN_SPAWN_GAP


# ------------------------------------------------------------------------------
# MARK: Task params.

MIN_GOAL_DIST = OBJ_RADIUS + BOT_WIDTH/2
MAX_GOAL_DIST = MIN_GOAL_DIST + 50.     # (ENV_DIAG_LENGTH - (WALL_WIDTH + BOT_RADIUS + OBJ_RADIUS)) * 2**0.5
GOAL_ZONE_RADIUS = 1.                   # MIN_GOAL_DIST + BOT_WIDTH|RADIUS*2
SIGNAL_RADIUS = GOAL_ZONE_RADIUS
GOAL_PRED_RADIUS = 4.                   # ((ENV_SIDE_LENGTH / 5)**2 / pi)**0.5

MAX_COORD_VAL = 18.                     # ENV_SIDE_LENGTH / 2
MAX_IMG_DEPTH = 24.                     # Longest unobstructed line of sight
MAX_RECOG_DIST = 8.                     # Distance at which an obj. has at least 4 pixels with a 96w-48h-90deg camera
MIN_GOAL_SPACING = 12.                  # max(ENV_SIDE_LENGTH / 3, GOAL_ZONE_RADIUS*2 + MIN_SPAWN_GAP)
ENV_HALFSPACING = 0.1

STEPS_PER_SECOND = 4
EP_DURATION = 90
VIS_EP_DURATION = 4

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
# TODO: Separate joint and individual reward
DISCOUNTS = (
    1.,
    1.,                                     # Long-horizon rewards: no half-life
    0.5 ** (1. / (3 * STEPS_PER_SECOND)))   # Short-horizon rewards: half-life at 3 seconds, gamma 0.944


# ------------------------------------------------------------------------------
# MARK: Colour palette

# 22 total: 2 background (sky, floor) + 9 wall + 8 goal/body + 1 chassis + 2 beacon (off, on)
COLOURS = {
    'background': (
        (230, 245, 255),    # Pale cyan
        (127, 127, 127)),   # Grey
    'wall': (
        (185, 255, 175),    # Light green
        (255, 255, 160),    # Yellow
        (255, 160, 205),    # Pink
        (255, 205, 135),    # Light orange
        (225, 160, 255),    # Violet
        (175, 245, 255),    # Light cyan
        (125, 235, 205),    # Turquoise
        (255, 215, 215),    # Pale pink
        (175, 195, 255)),   # Pale blue
    'goal': (
        (155, 205, 0),      # Yellow green
        (0, 145, 0),        # Green
        (255, 145, 0),      # Dark orange
        (0, 0, 255),        # Blue
        (0, 160, 200),      # Dark cyan
        (255, 0, 0),        # Red
        (100, 50, 0),       # Dark brown
        (160, 20, 150)),    # Dark magenta
    'chassis': (
        (102, 102, 102),),  # Grey
    'beacon': (
        (0, 0, 0),          # Black
        (255, 255, 255))}   # White

COLOURS = {clr_group: [[val / 255. for val in clr] for clr in clrs] for clr_group, clrs in COLOURS.items()}

N_WALL_CLRS = len(COLOURS['wall'])
N_GOAL_CLRS = len(COLOURS['goal'])
N_CLR_CLASSES = sum(len(clrs) for clrs in COLOURS.values())

SKY_CLR_IDX = 0
FLOOR_CLR_IDX = 1


# ------------------------------------------------------------------------------
# MARK: Visual entity classes

ENT_CLS_SKY = 0
ENT_CLS_FLOOR = 1
ENT_CLS_WALL = 2
ENT_CLS_OBJ = 3
ENT_CLS_BODY = 4
ENT_CLS_CHASSIS = 5
ENT_CLS_BEACON = 6
N_ENT_CLASSES = 7


# ------------------------------------------------------------------------------
# MARK: IO components

N_DOF_MOT = 4
N_DIM_RGB = 3
N_DIM_POS = 2
N_DIM_DIRLEN = 3
N_RCVR_PROX = 4
N_CARDINALS = 4

# 28 total:
# 4 torque cmd., 1 light cmd., 2 bot pos. xy, 2 global frame bot vel. xy,
# 2 accel. xy, 1 ang. vel. z, 2 sin-cos z, 4 prox. channels,
# 8 goal spec., 1 task completed, 1 time left until end of episode
OBS_VEC_SIZE = N_DOF_MOT + 1 + N_DIM_POS*3 + 3 + N_RCVR_PROX + N_GOAL_CLRS + 2
LED_ON_IDX = N_DOF_MOT

# Resolution mostly important for effective viewing distance
OBS_IMG_RES_WIDTH = 96
OBS_IMG_RES_HEIGHT = 48
OBS_IMG_CHANNELS = N_DIM_RGB + 1                                # RGB or HSV, depth (4)
ENC_IMG_CHANNEL_SPLIT = (OBS_IMG_CHANNELS, 2, 1)                # HSVD, clr. seg. + ent. seg., px. weights (7)
# TODO: Remove +1 from clr. classes after VisNet update
DEC_IMG_CHANNEL_SPLIT = (N_CLR_CLASSES+1, N_ENT_CLASSES, 1)     # Clr. softmax, ent. softmax, depth value (30)
ALL_IMG_CHANNEL_SPLIT = ENC_IMG_CHANNEL_SPLIT + DEC_IMG_CHANNEL_SPLIT

# Torque cmd., LED cmd.
ACT_SPLIT = (N_DOF_MOT, 1)

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
# 1 time left until end of episode, 1 avg. closest bot dist., 1 avg. vel. norm,
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

# 108 total:
# 28 obs., 50 common, 30 specific
STATE_VEC_SIZE = OBS_VEC_SIZE + STATE_ENV_SIZE-1 + STATE_BOT_SIZE
GOAL_FOUND_SLICE = slice(-6, -5)

# 15 total:
# 8 obj. presence, 4 NWSE cell walls, 1 cell clr. seg., 1 bot presence, 1 cell exploration
STATE_LAYOUT_CHANNELS = N_GOAL_CLRS + N_CARDINALS + 1
STATE_MAP_CHANNELS = STATE_LAYOUT_CHANNELS + 2

# 13 total:
# 8 obj. in frame + 1 for one-hot, 2 goal pos. xy + 2 xy std.
AUX_VAL_SPLIT = (N_GOAL_CLRS+1, N_DIM_POS*2)
AUX_VAL_SIZE = N_GOAL_CLRS + N_DIM_POS
AUX_VAL_OFFSET = OBS_VEC_SIZE + STATE_ENV_SIZE-1 + N_GOAL_CLRS
AUX_VAL_SLICE = slice(AUX_VAL_OFFSET, AUX_VAL_OFFSET + AUX_VAL_SIZE)


# ------------------------------------------------------------------------------
# MARK: Networks

COM_TOPIC = 2
COM_TARGET = 1
COM_NONE = 0

AUX_PRIO = 2
AUX_ONLINE = 1
AUX_NONE = 0

AGENT_TYPE_CONFIGS = {
    'main': {'com_mode': COM_TOPIC, 'aux_mode': AUX_PRIO},
    'tarmac': {'com_mode': COM_TARGET, 'aux_mode': AUX_PRIO},
    'silent': {'com_mode': COM_NONE, 'aux_mode': AUX_NONE},
    'pbl': {'com_mode': COM_TOPIC, 'aux_mode': AUX_ONLINE},
    'dial': {'com_mode': COM_TOPIC, 'aux_mode': AUX_NONE}}

K_DIM = 32
V_DIM = 64
QKV_SPLIT = (K_DIM, K_DIM, V_DIM)
N_TOPICS = 48

VIS_ENC_SIZE = 224
MAP_ENC_SIZE = 128
POL_ENC_SIZE = 320
VAL_ENC_SIZE = 512
VAL_PEN_SIZE = 256

MEM_SIZE = 256
CHRONO_RANGES = (1/STEPS_PER_SECOND, 1, 3, 10, 30)
CHRONO_SIZE = MEM_SIZE // (len(CHRONO_RANGES) - 1)


# ------------------------------------------------------------------------------
# MARK: Training setup

DATA_DIR = 'data'   # Training meta data and model checkpoints
LOG_DIR = 'runs'    # Tracked scalars

# Runner args.
N_ENVS = 9
N_BOTS = 112
SEEDS = (1, 3, 7, 9, 42)

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

# TODO: Separate schedules btw. network modules
# 2-30 virtual minutes to warm up, 8 hours per agent to train and anneal, almost a year over all 9*112 agents
UPDATE_MAP = {
    'policy': {'milestones': (10 * 90, 7 * 3600, 8 * 3600), 'lr': 1e-4},
    'critic': {'milestones': (5 * 90, 7 * 3600, 8 * 3600), 'lr': 3e-4},
    'vis-offline': {'milestones': (2 * 60, 2 * 3600, 8 * 3600), 'lr': 6e-4},
    'vis-online': {'milestones': (30 * 60, 7 * 3600, 8 * 3600), 'lr': 3e-5}}

N_EPOCHS_MAP = {k: tv['milestones'][-1] // SECONDS_PER_EPOCH for k, tv in UPDATE_MAP.items()}

UPDATE_MILESTONE_MAP = {
    k: tuple([v * N_UPDATES_PER_EPOCH // SECONDS_PER_EPOCH for v in tv['milestones']])
    for k, tv in UPDATE_MAP.items()}

WEIGHT_DECAY = 1e-4
VALUE_WEIGHT = 1.
AUX_WEIGHT = 1e-2
ENT_WEIGHT_MILESTONES = (1e-2, 1e-3)
