"""Command caller with partially preset arguments"""

import os
from argparse import ArgumentParser


ARG_CFG = {
    'train_vis': 'python src/mazebots/train_vis.py',
    'maze': 'python src/mazebots/maze.py'}

for i in range(1, 8):
    # NOTE: May need to add `DISPLAY=:0.0` to get rid of seg. faults on a headless server
    base_call = f'DRI_PRIME=1 python src/mazebots/session.py --level {i}'

    ARG_CFG[f'test{i}'] = base_call

    # NOTE: Gets 64x256 (16k) images, or about 1.5 GB at dims. 96x48x5 per level
    # Bots and goals are repositioned and recoloured on every step before collecting
    ARG_CFG[f'collect{i}'] = f'{base_call} --ctrl_mode 1 --headless 1 --rec_mode 2 --x_duration 0 --end_step 64'

    ARG_CFG[f'train{i}'] = f'{base_call} --ctrl_mode 2 --headless 1'
    ARG_CFG[f'eval{i}'] = f'{base_call} --ctrl_mode 3'


if __name__ == '__main__':
    parser = ArgumentParser(description='MazeBots runner.')

    parser.add_argument('-m', '--mode', type=str, default='test4', help='Call specification.')
    parser.add_argument('-a', '--args', type=str, default='', help='Arguments to relay.')

    # Compensate libpython package error
    # NOTE: This should be changed for other setups
    parser.add_argument(
        '-e', '--exe', type=str,
        default='export LD_LIBRARY_PATH=/home/jpuc/miniconda3/envs/rl2/lib',
        help='Command to execute before the main call.')

    parsed_args = parser.parse_args()
    relayed_arg_str: str = parsed_args.args
    pre_exe: str = parsed_args.exe
    call_mode: str = parsed_args.mode

    if call_mode not in ARG_CFG:
        raise KeyError('Call mode not recognised.')

    if pre_exe:
        print(f'Running: {pre_exe}')
        exit_flag = os.system(pre_exe)

        if exit_flag:
            raise RuntimeError(f'Unable to execute command before the main call: {pre_exe}')

    sys_call = ARG_CFG[call_mode] + (f' {relayed_arg_str}' if relayed_arg_str else '')

    print(f'Running: {sys_call}')
    exit_flag = os.system(sys_call)

    if exit_flag == 0:
        print('Process finished successfully.')

    elif exit_flag == 2:
        print('Process aborted by user command.')

    else:
        print(f'Process errored out with exit flag {exit_flag}.')
