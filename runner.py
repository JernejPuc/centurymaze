"""Command caller with partially preset arguments"""

import os
from argparse import ArgumentParser
from time import sleep

import src.mazebots.config as cfg_proto
import src.mazebotsgen.config as cfg_gen
import src.mazebotstex.config as cfg_tex


# ------------------------------------------------------------------------------
# MARK: V1 - proto

V1_RUN_CALL = 'python src/mazebots/session.py'
# NOTE: V1_REMODEL_CALL not implemented (manual)

V1_PRESETS: 'dict[str, list[str]]' = {
    'test': [
        V1_RUN_CALL],
    'train_vis': [
        f'{V1_RUN_CALL} --n_envs 1 --n_bots {cfg_proto.N_BOTS} --ep_duration {cfg_proto.VIS_EP_DURATION} '
        '--ctrl_mode 1 --headless 1 --global_spawn_prob 0.95'],
    'train_rl': [
        f'{V1_RUN_CALL} --n_envs {cfg_proto.N_ENVS} --n_bots {cfg_proto.N_BOTS} '
        '--ctrl_mode 2 --headless 1'],
    'train_all': [],
    'eval': [
        V1_RUN_CALL + ' --ctrl_mode 3'],
    'eval_perf': [
        f'{V1_RUN_CALL} --n_bots {cfg_proto.N_BOTS} --end_step {cfg_proto.N_EVAL_STEPS} '
        '--ctrl_mode 3 --rec_mode 2 --headless 1']}

for seed in cfg_proto.SEEDS:
    for agent_type, args in cfg_proto.AGENT_TYPE_CONFIGS.items():
        V1_PRESETS['train_all'].append(
            f'{V1_PRESETS["train_rl"][0]} --model_name {agent_type}{seed} --rng_seed {seed} '
            f'{" ".join(f"--{k} {v}" for k, v in args.items())}')


# ------------------------------------------------------------------------------
# MARK: V2 - gen

V2_RUN_CALL = 'python src/mazebotsgen/session.py'
V2_REMODEL_CALL = 'python src/mazebotsgen/model.py'

V2_PRESETS = {
    'test': [
        V2_RUN_CALL],
    'train_vis': [
        f'{V2_RUN_CALL} --env_cfg prog --ep_duration {cfg_gen.VIS_EP_DURATION} '
        '--ctrl_mode 1 --headless 1 --global_spawn_prob 0.95'],
    'train_rl': [
        V2_RUN_CALL + ' --env_cfg prog --ctrl_mode 2 --headless 1'],
    'eval': [
        V2_RUN_CALL + ' --ctrl_mode 3'],
    'eval_perf': []}

V2_PRESETS['train_all'] = [
    V2_PRESETS['train_vis'][0] + ' --model_name visgen',
    V2_REMODEL_CALL + ' --model_name visgen']

for seed in cfg_gen.SEEDS:
    for agent_type, args in cfg_gen.AGENT_TYPE_CONFIGS.items():
        V2_PRESETS['train_all'].append(
            f'{V2_PRESETS["train_rl"][0]} --model_name {agent_type}{seed} --rng_seed {seed} '
            f'{" ".join(f"--{k} {v}" for k, v in args.items())}')

for lvl, steps in cfg_gen.LEVEL_EVAL_STEPS.items():
    V2_PRESETS['eval_perf'].append(
        f'{V2_PRESETS["eval"][0]} --rec_mode 2 --headless 1 --env_cfg 1x{lvl} --end_step {steps}')


# ------------------------------------------------------------------------------
# MARK: V3 - tex

V3_RUN_CALL = 'python src/mazebotstex/session.py'
V3_REMODEL_CALL = 'python src/mazebotstex/model.py'

V3_PRESETS = {
    'test': [
        V3_RUN_CALL],
    'train_vis': [
        f'{V3_RUN_CALL} --env_cfg vis --ep_duration {cfg_tex.VIS_EP_DURATION} '
        '--ctrl_mode 1 --headless 1 --global_spawn_prob 0.95 --aug_num 100'],
    'train_rl': [
        V3_RUN_CALL + ' --env_cfg prog --ctrl_mode 2 --headless 1 --aug_num 100'],
    'eval': [
        V3_RUN_CALL + ' --ctrl_mode 3'],
    'train_all': [],
    'eval_perf': []}

for enc_mode in (cfg_tex.TEXT_NONE, cfg_tex.TEXT_MIXED):
    V3_PRESETS['train_all'].extend([
        f'{V3_PRESETS["train_vis"][0]} --model_name vistex{enc_mode} --text_mode {enc_mode}',
        f'{V3_REMODEL_CALL} --model_name vistex{enc_mode} '
        f'--out_dir {os.path.join(cfg_tex.DATA_DIR, f"vistex{enc_mode}")}'])

for enc_mode, pi_mode in (
    (cfg_tex.TEXT_NONE, cfg_tex.TEXT_NONE),
    (cfg_tex.TEXT_MIXED, cfg_tex.TEXT_NONE),
    (cfg_tex.TEXT_MIXED, cfg_tex.TEXT_MIXED)
):
    V3_PRESETS['train_all'].append(
        f'{V3_PRESETS["train_rl"][0]} --model_name diablx{enc_mode}{pi_mode} --text_mode {pi_mode} '
        f'--visnet_dir {os.path.join(cfg_tex.DATA_DIR, f"vistex{enc_mode}")} '
        f'{" ".join(f"--{k} {v}" for k, v in cfg_tex.AGENT_TYPE_CONFIGS["diabl-infer"].items())}')

V3_PRESETS['train_all'].extend([
    f'{V3_PRESETS["train_vis"][0]} --model_name vistex{cfg_tex.TEXT_SPLIT} --text_mode {cfg_tex.TEXT_SPLIT} '
    f'--visnet_dir {os.path.join(cfg_tex.DATA_DIR, f"vistex{cfg_tex.TEXT_NONE}")}',
    f'{V3_REMODEL_CALL} --visnet_name vistex{cfg_tex.TEXT_SPLIT} '
    f'--policy_name diablx{cfg_tex.TEXT_NONE}{cfg_tex.TEXT_NONE} '
    f'--new_name diablx{cfg_tex.TEXT_NONE}{cfg_tex.TEXT_SPLIT}'])

for enc_mode, pi_mode in (
    (cfg_tex.TEXT_NONE, cfg_tex.TEXT_NONE),
    (cfg_tex.TEXT_NONE, cfg_tex.TEXT_SPLIT),
    (cfg_tex.TEXT_MIXED, cfg_tex.TEXT_NONE),
    (cfg_tex.TEXT_MIXED, cfg_tex.TEXT_MIXED)
):
    for lvl, steps in cfg_tex.LEVEL_EVAL_STEPS.items():
        for t in (cfg_tex.TEXT_NONE, cfg_tex.TEXT_WARES, cfg_tex.TEXT_MIXED):
            V3_PRESETS['eval_perf'].append(
                f'{V3_PRESETS["eval"][0]} --model_name diablx{enc_mode}{pi_mode} --env_cfg 1x{lvl} --end_step {steps} '
                f'--text_mode {t} --aug_num 100 --rec_mode 2 --headless 1')


# ------------------------------------------------------------------------------
# MARK: Runner

CMD_PRESETS = {'proto': V1_PRESETS, 'gen': V2_PRESETS, 'tex': V3_PRESETS}


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

    for i, sys_call in enumerate(preset, start=1):
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

        if i != len(preset):
            print()

            for i in range(5, 0, -1):
                print(f'\rStarting next run in {i}...', end='')
                sleep(1)

            print()
