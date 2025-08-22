"""Command caller with partially preset arguments"""

import os
from argparse import ArgumentParser
from time import sleep

from src.mazebots.config import AGENT_TYPE_CONFIGS, N_BOTS, N_ENVS, N_EVAL_STEPS, SEEDS, VIS_EP_DURATION
from src.mazebotsgen.config import AGENT_TYPE_CONFIGS as AGENT_CFG_2, VIS_EP_DURATION as VIS_DUR_2, LEVEL_EVAL_STEPS


V1_CALL = 'python src/mazebots/session.py'

V1_PRESETS: 'dict[str, list[str]]' = {
    'test': [V1_CALL],
    'train_vis': [
        f'{V1_CALL} --n_bots {N_BOTS} --ep_duration {VIS_EP_DURATION} '
        '--ctrl_mode 1 --headless 1 --n_envs 1 --global_spawn_prob 0.95'],
    'train_rl': [f'{V1_CALL} --ctrl_mode 2 --headless 1 --n_envs {N_ENVS} --n_bots {N_BOTS}'],
    'train_all': [],
    'eval': [V1_CALL + ' --ctrl_mode 3'],
    'eval_perf': [f'{V1_CALL} --ctrl_mode 3 --rec_mode 2 --headless 1 --n_bots {N_BOTS} --end_step {N_EVAL_STEPS}']}

for seed in SEEDS:
    for agent_type, args in AGENT_TYPE_CONFIGS.items():
        V1_PRESETS['train_all'].append(
            f'{V1_PRESETS["train_rl"][0]} --model_name {agent_type}{seed} --rng_seed {seed} '
            f'{" ".join(f"--{k} {v}" for k, v in args.items())}')

V2_CALL = 'python src/mazebotsgen/session.py'

V2_PRESETS = {
    'test': [V2_CALL],
    'train_vis': [
        f'{V2_CALL} --env_cfg prog --ep_duration {VIS_DUR_2} '
        '--ctrl_mode 1 --headless 1 --global_spawn_prob 0.95'],
    'train_rl': [V2_CALL + ' --env_cfg prog --ctrl_mode 2 --headless 1'],
    'eval': [V2_CALL + ' --ctrl_mode 3'],
    'eval_perf': []}

V2_PRESETS['train_all'] = [
    V2_PRESETS['train_vis'][0] + ' --model_name visgen',
    'python src/mazebotsgen/model.py --model_name visgen']

for seed in SEEDS:
    for agent_type, args in AGENT_CFG_2.items():
        V2_PRESETS['train_all'].append(
            f'{V2_PRESETS["train_rl"][0]} --model_name {agent_type}{seed} --rng_seed {seed} '
            f'{" ".join(f"--{k} {v}" for k, v in args.items())}')

for lvl, steps in LEVEL_EVAL_STEPS.items():
    V2_PRESETS['eval_perf'].append(
        f'{V2_PRESETS["eval"][0]} --rec_mode 2 --headless 1 --env_cfg 1x{lvl} --end_step {steps}')

CMD_PRESETS = {'proto': V1_PRESETS, 'gen': V2_PRESETS}


if __name__ == '__main__':
    parser = ArgumentParser(description='MazeBots runner.')

    parser.add_argument(
        '-f', '--fix_headless', action='store_true', help='Add env. var. to avoid seg. faults on a headless server.')
    parser.add_argument('-p', '--prefix', type=str, default='DRI_PRIME=1', help='Call prefix, e.g. GPU config. var.')
    parser.add_argument('-v', '--version', type=str, default='gen', help='Maze/task version.')
    parser.add_argument('-s', '--spec', type=str, default='test', help='Preset specification.')
    parser.add_argument('-a', '--args', type=str, default='', help='Additional arguments to relay.')
    parser.add_argument('-g', '--gpu_id', type=int, default=-1, help='GPU device num. id.')

    parsed = parser.parse_args()

    presets = CMD_PRESETS.get(parsed.version)
    preset_spec: str = parsed.spec

    if presets is None:
        raise KeyError(f'Version "{parsed.version}" not recognised.')

    if preset_spec not in presets:
        raise KeyError(f'Preset "{preset_spec}" not recognised.')

    preset = presets[preset_spec]

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
