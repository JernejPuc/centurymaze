"""Command caller with partially preset arguments"""

import os
from argparse import ArgumentParser
from time import sleep


CMD_PRESETS: 'dict[str, list[str]]' = {
    'train_vis': ['python src/mazebots/train_vis.py'],
    'maze': ['python src/mazebots/maze.py']}


# Base
for i in range(1, 8):
    base_call = f'python src/mazebots/session.py --level {i}'

    CMD_PRESETS[f'test{i}'] = [base_call]

    # NOTE: Gets 64x256 (16k) images, or about 1.5 GB at dims. 96x48x5 per level
    # Bots and goals are repositioned and recoloured on every step before collecting
    CMD_PRESETS[f'collect{i}'] = [f'{base_call} --ctrl_mode 1 --headless 1 --rec_mode 2 --mul_duration 0 --end_step 64']

    CMD_PRESETS[f'train{i}'] = [f'{base_call} --ctrl_mode 2 --headless 1']
    CMD_PRESETS[f'eval{i}'] = [f'{base_call} --ctrl_mode 3']


# Curriculum
AGENT_TYPE_CONFIGS = {
    'base': {'transfer': None, 'com_state': 0, 'guide_state': 3, 'com_ref': 0.1, 'n_speakers': 0},
    'orac': {'transfer': None, 'com_state': 0, 'guide_state': 1, 'com_ref': 1., 'n_speakers': 0},
    'hear': {'transfer': None, 'com_state': 1, 'guide_state': 3, 'com_ref': 1., 'n_speakers': -1},
    'free': {'transfer': None, 'com_state': 2, 'guide_state': 3, 'com_ref': 0.1, 'n_speakers': -1},
    'heur': {'transfer': 'hear', 'com_state': 1, 'guide_state': 3, 'com_ref': 0.1, 'n_speakers': -1},
    'cond': {'transfer': 'hear', 'com_state': 3, 'guide_state': 3, 'com_ref': 0.1, 'n_speakers': ...}}

N_ENVS = 8
N_BOTS = 256

for seed in (0, 42, 100):
    for agent_type, args in AGENT_TYPE_CONFIGS.items():
        cmd_seq = []

        transfer_key = args['transfer']
        model_name = '' if transfer_key is None else f'{transfer_key}{seed}v7_{N_ENVS}e-128a-4m'

        speaker_curriculum = args['n_speakers'] is ...
        speaker_phase = speaker_curriculum and transfer_key is not None

        levels = (((2,)*9 + (3, 4)) if speaker_phase else ((5,)*11)) + (5, 5, 5, 6, 7)
        durations = (0.25 if speaker_phase else 1,)*11 + (1, 1.5, 2, 3, 4)

        for stage, (level, ep_duration) in enumerate(zip(levels, durations)):
            n_bots = 2 ** level
            n_speakers = 2 ** max(0, stage - 8) if speaker_curriculum else args['n_speakers']
            n_envs = max(N_ENVS, N_BOTS // n_bots)
            n_objects = -1

            com_state = 2 if speaker_phase and stage >= 11 else args['com_state']
            guide_state, com_ref = args['guide_state'], args['com_ref']

            clr_idx = stage if stage < 8 else -1
            com_spawn = int(speaker_phase and stage < 11)

            if com_spawn:
                n_objects = level
                level = 7

            schedule_key = 'default' if stage < 11 else f'{N_ENVS}e-{n_bots}a-{int(ep_duration)}m'

            if level == 7:
                mul_duration = ep_duration / 4.5

            elif level == 5:
                mul_duration = ep_duration / 2.

            else:
                mul_duration = 1.

            transfer_name = model_name
            model_name = f'{agent_type}{seed}v{stage}_{schedule_key}'

            cmd_seq.append(
                f'python src/mazebots/session.py --level {level} --n_envs {n_envs} --ctrl_mode 2 --headless 1 '
                f'--n_bots {n_bots} --n_objects {n_objects} --n_colours 8 --clr_idx {clr_idx} '
                f'--model_name "{model_name}" --transfer_name "{transfer_name}" --rng_seed {seed} '
                f'--n_speakers {n_speakers} --mul_duration {mul_duration} --schedule_key "{schedule_key}" '
                f'--com_state {com_state} --guide_state {guide_state} --com_spawn {com_spawn} --com_ref {com_ref}')

        CMD_PRESETS[f'curr_{agent_type}{seed}'] = cmd_seq


if __name__ == '__main__':
    parser = ArgumentParser(description='MazeBots runner.')

    parser.add_argument(
        '-f', '--fix_headless', action='store_true', help='Add env. var. to avoid seg. faults on a headless server.')
    parser.add_argument('-p', '--prefix', type=str, default='DRI_PRIME=1', help='Call prefix, e.g. GPU config. var.')
    parser.add_argument('-s', '--spec', type=str, default='test4', help='Preset specification.')
    parser.add_argument('-a', '--args', type=str, default='', help='Additional arguments to relay.')
    parser.add_argument('-g', '--gpu_id', type=int, default=-1, help='GPU device num. id.')

    # NOTE: This should be changed for other setups
    parser.add_argument(
        '-l', '--lib_path', type=str,
        default='/home/jpuc/miniconda3/envs/rl2/lib',
        help='Explicit python env. path to set as env. var. and resolve a libpython package error.')

    parsed = parser.parse_args()

    preset_spec: str = parsed.spec
    lib_path: str = parsed.lib_path

    if preset_spec not in CMD_PRESETS:
        raise KeyError(f'Preset "{preset_spec}" not recognised.')

    if lib_path:
        sys_call = f'export LD_LIBRARY_PATH={lib_path}'
        exit_flag = os.system(sys_call)

        if exit_flag:
            raise RuntimeError(f'Unable to execute: {sys_call}')

    preset = CMD_PRESETS[preset_spec]

    prefix: str = (
        ('DISPLAY=:0.0' if parsed.fix_headless else '') +
        (' ' if parsed.fix_headless and parsed.prefix else '') +
        parsed.prefix + ' ')

    suffix: str = (
        (' ' if parsed.gpu_id >= 0 or parsed.args else '') +
        ('' if parsed.gpu_id < 0 else f'--sim_device "cuda:{parsed.gpu_id}" --graphics_device_id {parsed.gpu_id} ') +
        parsed.args)

    for sys_call in preset:
        sys_call = prefix + sys_call + suffix

        print(f'Running: {sys_call}')
        exit_flag = os.system(sys_call)

        if exit_flag == 0:
            print('Process finished successfully.')

        elif exit_flag == 2:
            print('Process aborted by user command.')
            break

        elif exit_flag == 35584:
            print(
                'Process probably encountered a display-related segmentation fault with no adverse consequences. '
                'See the `fix_headless` arg. to avoid it in the future.')

        else:
            print(f'Process errored out with exit flag {exit_flag}.')
            break

        if len(preset) > 1:
            print()

            for i in range(5, 0, -1):
                print(f'\rStarting next run in {i}...', end='')
                sleep(1)

            print()
