"""Command caller with partially preset arguments"""

import os
from argparse import ArgumentParser
from time import sleep

from src.mazebots.config import AGENT_TYPE_CONFIGS, N_BOTS, N_ENVS, SEEDS, VIS_EP_DURATION


BASE_CALL = 'python src/mazebots/session.py'

CMD_PRESETS: 'dict[str, list[str]]' = {
    'test': [BASE_CALL],
    'train_vis': [
        f'{BASE_CALL} --n_bots {N_BOTS} --ep_duration {VIS_EP_DURATION} '
        '--ctrl_mode 1 --headless 1 --n_envs 1 --global_spawn_prob 0.95'],
    'train_rl': [f'{BASE_CALL} --ctrl_mode 2 --headless 1 --n_envs {N_ENVS} --n_bots {N_BOTS}'],
    'eval': [BASE_CALL + ' --ctrl_mode 3']}

CMD_PRESETS['train_all'] = []

for seed in SEEDS:
    for agent_type, args in AGENT_TYPE_CONFIGS.items():
        CMD_PRESETS['train_all'].append(
            f'{CMD_PRESETS["train_rl"][0]} '
            f'--model_name {agent_type}{seed} --rng_seed {seed} {" ".join(f"--{k} {v}" for k, v in args.items())}')


if __name__ == '__main__':
    parser = ArgumentParser(description='MazeBots runner.')

    parser.add_argument(
        '-f', '--fix_headless', action='store_true', help='Add env. var. to avoid seg. faults on a headless server.')
    parser.add_argument('-p', '--prefix', type=str, default='DRI_PRIME=1', help='Call prefix, e.g. GPU config. var.')
    parser.add_argument('-s', '--spec', type=str, default='test', help='Preset specification.')
    parser.add_argument('-a', '--args', type=str, default='', help='Additional arguments to relay.')
    parser.add_argument('-g', '--gpu_id', type=int, default=-1, help='GPU device num. id.')

    parsed = parser.parse_args()

    preset_spec: str = parsed.spec

    if preset_spec not in CMD_PRESETS:
        raise KeyError(f'Preset "{preset_spec}" not recognised.')

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
