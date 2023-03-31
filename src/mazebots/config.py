"""Project-wide configuration"""

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
IMU_VEC_SIZE = 3*3
RGB_VEC_SIZE = 3
REC_VEC_SIZE = RGB_VEC_SIZE*3
DIR_VEC_SIZE = 2

# 6 total: 2 air xy direction, 2 a* xy direction, 1 air distance, 1 a* path length
GUIDE_VEC_SIZE = 2*DIR_VEC_SIZE + 2

# 34 total:
# 1 confirmation status (time) at goal, 1 throughput,
# 4 torque cmd., 4 ang. vel., 3x3 IMU,
# 3 RGB cmd., 3 RGB task, 3x3 RGB directed receivers
OBS_VEC_SIZE = 2 + 2*DOF_VEC_SIZE + IMU_VEC_SIZE + 2*RGB_VEC_SIZE + REC_VEC_SIZE

# Resolution mostly important for effective viewing distance
OBS_IMG_RES_WIDTH = 96
OBS_IMG_RES_HEIGHT = 48
OBS_IMG_CHANNELS = RGB_VEC_SIZE + 1  # RGB, depth

# Torque cmd., rgb radial emitter
ACT_VEC_SIZE = DOF_VEC_SIZE + RGB_VEC_SIZE
ACT_VEC_SPLIT = (DOF_VEC_SIZE, RGB_VEC_SIZE)

# 12 total:
# 6 guide vec., 1 line of sight flag, 1 time at goal, 1 time spent on task,
# 1 time until end of episode, 1 num. of completed tasks, 1 throughput
STATE_VEC_SIZE = GUIDE_VEC_SIZE + 6

# Training setup
DATA_DIR = 'data'   # Training meta data and model checkpoints
LOG_DIR = 'runs'    # Tracked scalars

# GAE length etc.
STEPS_PER_SECOND = 4
N_TRUNCATED_STEPS = STEPS_PER_SECOND * 4
N_ROLLOUT_STEPS = 256
N_ROLLOUTS_PER_EPOCH = 8
N_AUX_ITERS_PER_EPOCH = 6
SECONDS_PER_EPOCH = N_ROLLOUT_STEPS * N_ROLLOUTS_PER_EPOCH // STEPS_PER_SECOND

# Main + aux. updates
N_UPDATES_PER_EPOCH = (
    N_ROLLOUT_STEPS // N_TRUNCATED_STEPS * N_ROLLOUTS_PER_EPOCH
    + N_ROLLOUT_STEPS * N_ROLLOUTS_PER_EPOCH // N_TRUNCATED_STEPS * N_AUX_ITERS_PER_EPOCH)

# Approx. 1 plot point per 10 virtual minutes, 8 per hour, 192 per day, 1344 per week
LOG_EPOCH_INTERVAL = 10 * 60 // SECONDS_PER_EPOCH

# Approx. 1 checkpoint per half virtual hour, 48 per day, 336 per week
CKPT_EPOCH_INTERVAL = 30 * 60 // SECONDS_PER_EPOCH

# Approx. 1 branch per 2 virtual hours, 12 per day, 84 per week
BRANCH_EPOCH_INTERVAL = 120 * 60 // SECONDS_PER_EPOCH

# TODO: Adjust wrt. trials
TIME_MILESTONE_MAP = {
    # Approx. half virtual hour to warm up, half day to train, half day to cool down, 1 day total
    1: (1800, 12 * 3600, 24 * 3600),
    # Approx. 1 virtual hour to warm up, 5 days to train, 2 days to cool down, 1 week total
    4: (3600, 5 * 24 * 3600, 7 * 24 * 3600)}

N_EPOCHS_MAP = {k: tv[-1] // SECONDS_PER_EPOCH for k, tv in TIME_MILESTONE_MAP.items()}

UPDATE_MILESTONE_MAP = {
    k: tuple([v * N_UPDATES_PER_EPOCH // SECONDS_PER_EPOCH for v in tv])
    for k, tv in TIME_MILESTONE_MAP.items()}
