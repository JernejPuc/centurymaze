"""Pre-train MazeBots' visual encoder on a collection of env. images"""

import os
from argparse import ArgumentParser, Namespace
from datetime import timedelta
from time import perf_counter

import torch
from torch import Tensor
from torch.nn import functional
from torch.utils.tensorboard import SummaryWriter

from accel import build_graph
from config import DATA_DIR, LOG_DIR, N_SEG_CLASSES
from model import VisNet
from utils_train import CheckpointTracker, LoadedDataset, SideLoadingDataset, NAdamW, SoftConstLRScheduler


class Trainer:
    MAX_DISP_SECONDS = 99*24*3600

    def __init__(self, args: Namespace):

        # Unpack args
        model_name: str = args.model_name
        device: str = args.device
        rng_seed: int = args.rng_seed

        self.n_epochs: int = args.n_epochs
        self.log_interval: int = args.log_epoch_interval
        self.checkpoint_interval: int = args.ckpt_epoch_interval
        self.branch_interval: int = args.branch_epoch_interval

        lr_main_start_epoch: int = args.lr_main_start_epoch
        lr_main_end_epoch: int = args.lr_main_end_epoch

        file_name: str = args.file_name
        sideload: bool = args.sideload
        batch_size: int = args.batch_size
        self.mirror_prob: float = args.mirror_prob

        # Init. model
        if device.startswith('cuda'):
            torch.backends.cudnn.benchmark = True

        self.ckpter = CheckpointTracker(model_name, DATA_DIR, device, rng_seed)

        self.model = VisNet()
        self.optimiser = NAdamW(self.model.parameters(), lr=2e-5, weight_decay=0.)
        self.ckpter.load_model(self.model, self.optimiser)

        self.scheduler = SoftConstLRScheduler(
            self.optimiser,
            step_milestones=[lr_main_start_epoch, lr_main_end_epoch, self.n_epochs],
            starting_step=self.ckpter.meta['update_step'])

        # Logging
        self.loss = torch.tensor(0., device=device)
        new_zero_tensor = self.loss.clone

        self.stats = {
            'loss': new_zero_tensor(),
            'rgb_wmae': new_zero_tensor(),
            'dep_wmse': new_zero_tensor(),
            'seg_fce': new_zero_tensor()}

        self.writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, model_name))
        self.write = self.writer.add_scalar

        # Dataset
        self.dataset = (
            SideLoadingDataset
            if device.startswith('cuda') and sideload
            else LoadedDataset)(
                os.path.join(DATA_DIR, file_name), batch_size, device)

        self.n_steps_per_epoch = max(1, len(self.dataset) // batch_size)

        # Accelerate update step
        if device.startswith('cuda'):

            # Warmup and reset gradients
            ref_batches = self.dataset.data[:batch_size*4].to(device).chunk(4)

            for batch in ref_batches[:3]:
                self.optimiser.zero_grad(set_to_none=True)
                self.update(batch)

            # Capture computational graph
            self.optimiser.zero_grad(set_to_none=True)

            self.update, self.graph = build_graph(
                self.update,
                ref_batches[3].clone(),
                warmup_tensor_list=(),
                single_input=True)

        else:
            self.graph = None

    def run(self):
        starting_step = self.ckpter.meta['epoch_step']
        starting_time = perf_counter()

        for epoch_step in range(starting_step+1, self.n_epochs+1):

            # Estimate time remaining
            progress = epoch_step / self.n_epochs
            running_time = perf_counter() - starting_time

            remaining_time = min(
                int(running_time * (self.n_epochs - epoch_step) / (epoch_step - starting_step)),
                self.MAX_DISP_SECONDS)

            for iter_step, batch in enumerate(self.dataset, start=1):
                self.print_progress(progress, remaining_time, epoch_step, iter_step)

                # Mirror batch
                if self.ckpter.rng.random() < self.mirror_prob:
                    batch = batch.flip(-1)

                if self.graph is None:
                    self.optimiser.zero_grad()

                self.update(batch)
                self.scheduler.step()

            # Log running metrics and perf. score
            if self.log_interval and not epoch_step % self.log_interval:
                self.log(epoch_step)

            # Save model params. and training state
            if self.branch_interval and not epoch_step % self.branch_interval:
                self.checkpoint(epoch_step, branch=True)

            elif self.checkpoint_interval and not epoch_step % self.checkpoint_interval:
                self.checkpoint(epoch_step)

        self.checkpoint(epoch_step)

    def print_progress(
        self,
        progress: float,
        remaining_time: float,
        epoch_step: int,
        iter_step: int
    ):
        print(
            f'\rEpoch {epoch_step} of {self.n_epochs} ({progress:.2f}) | '
            f'Iter. {iter_step} of {self.n_steps_per_epoch} | '
            f'ETA: {str(timedelta(seconds=remaining_time))} | '
            f'Loss: {self.loss.item() / (iter_step ):.4f}        ',
            end='')

        self.loss.zero_()

    def log(self, epoch_step: int):
        den = self.n_steps_per_epoch * self.log_interval

        for key, val in tuple(self.stats.items()):
            self.write(key, val.item() / den, epoch_step)
            self.stats[key].zero_()

    def checkpoint(self, epoch_step: int, branch: bool = False):
        update_step = self.scheduler.step_ctr
        ckpt_increment = 1 if branch else 0

        self.ckpter.checkpoint(epoch_step, update_step, ckpt_increment, self.loss.item())

    def update(self, targets: Tensor):
        rgbdep, seg = self.model(targets[:, :4])

        # Split RGB and depth
        rgb = rgbdep[:, :3]
        dep = rgbdep[:, 3]

        # Logits to probs.
        log_prob = functional.log_softmax(seg, dim=1)
        prob = functional.softmax(seg, dim=1)

        rgb_target = targets[:, :3]
        dep_target = targets[:, 3]
        seg_target = targets[:, 4].long()
        px_weights = targets[:, 5:]

        # Expand segmentation to probs. (one-hot)
        prob_target = functional.one_hot(seg_target, N_SEG_CLASSES).neg_()
        prob_target = torch.movedim(prob_target, -1, 1)

        # Weighted mean absolute error (external weights)
        rgb_wmae = ((rgb - rgb_target).abs() * px_weights).mean()

        # Weighted mean squared error
        dep_mse = ((dep - dep_target).square() * px_weights).mean()

        # Focal cross entropy (gamma 1)
        seg_fce = ((1. - prob) * (log_prob * prob_target)).mean()

        # Update params.
        loss = rgb_wmae + dep_mse + seg_fce
        loss.backward()

        self.optimiser.step()

        # Stats for logging
        with torch.no_grad():

            # NOTE: Loss is later divided by iter_step, hence noisy at the start of each epoch
            self.loss += loss

            self.stats['loss'] += loss
            self.stats['rgb_wmae'] += rgb_wmae
            self.stats['dep_wmse'] += dep_mse
            self.stats['seg_fce'] += seg_fce


if __name__ == '__main__':
    parser = ArgumentParser(description='Vis. enc. pre-training.')

    parser.add_argument(
        '--model_name', type=str, default='visnet',
        help='Name under which model checkpoints and events will be saved.')
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='Device to train on (CPU, GPU, or GPU with index).')
    parser.add_argument(
        '--rng_seed', type=int, default=42,
        help='Seed for initialising random number generators.')

    # TODO: Adjust wrt. trials
    parser.add_argument(
        '--n_epochs', type=int, default=2000,
        help='Number of repeated iterations through the dataset.')
    parser.add_argument(
        '--log_epoch_interval', type=int, default=10,
        help='Interval for logging current loss components.')
    parser.add_argument(
        '--ckpt_epoch_interval', type=int, default=250,
        help='Interval for saving current model parameters.')
    parser.add_argument(
        '--branch_epoch_interval', type=int, default=500,
        help='Interval for starting a new branch, i.e. path to save current model parameters.')

    parser.add_argument(
        '--lr_main_start_epoch', type=float, default=1000,
        help='Epoch on which the learning rate assumes its peak value.')
    parser.add_argument(
        '--lr_main_end_epoch', type=float, default=1500,
        help='Epoch on which the learning rate proceeds to cool down.')

    parser.add_argument(
        '--file_name', type=str, default='dataset.pt',
        help='Name of the file with pre-processed image data.')
    parser.add_argument(
        '--sideload', type=int, default=0,
        help='Option to stream batches from RAM instead of moving the whole dataset to target device.')
    parser.add_argument(
        '--batch_size', type=int, default=64,
        help='Number of images processed simultaneously per update.')
    parser.add_argument(
        '--mirror_prob', type=float, default=0.1,
        help='Probability of flipping images in a batch over the horizontal dimension.')

    args = parser.parse_args()
    trainer = Trainer(args)

    try:
        trainer.run()

    except KeyboardInterrupt:
        print('\nTraining interrupted by user.', end='')

    print('\nDone.')
