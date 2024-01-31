"""Pre-train MazeBots' visual encoder on a collection of env. images"""

import json
import os
from argparse import ArgumentParser, Namespace
from datetime import timedelta
from time import perf_counter

import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import log_softmax, one_hot
from torch.utils.tensorboard import SummaryWriter

from discit.accel import capture_graph
from discit.data import LoadedDataset, SideLoadingDataset
from discit.optim import NAdamW, AnnealingScheduler
from discit.track import CheckpointTracker

import config as cfg
from model import VisNet


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

        self.ckpter = CheckpointTracker(model_name, cfg.DATA_DIR, device, rng_seed)

        self.model = VisNet()
        self.optimizer = NAdamW(self.model.parameters(), lr=args.lr_max, weight_decay=1e-2, device=device)
        self.ckpter.load_model(self.model, self.optimizer)

        self.scheduler = AnnealingScheduler(
            self.optimizer,
            step_milestones=[lr_main_start_epoch, lr_main_end_epoch, self.n_epochs],
            lr_div_factors=[args.lr_max/args.lr_init, args.lr_max/args.lr_final],
            starting_step=self.ckpter.meta['update_step'])

        # Logging
        self.loss = torch.tensor(0., device=device)
        new_zero_tensor = self.loss.clone

        self.stats = {
            'visnet/loss': new_zero_tensor(),
            'visnet/dep_mse': new_zero_tensor(),
            'visnet/clr_fce': new_zero_tensor(),
            'visnet/typ_fce': new_zero_tensor()}

        self.writer = SummaryWriter(log_dir=os.path.join(cfg.LOG_DIR, model_name))
        self.write = self.writer.add_scalar

        # Dataset
        self.dataset = (
            SideLoadingDataset
            if device.startswith('cuda') and sideload
            else LoadedDataset)(
                np.load(os.path.join(cfg.DATA_DIR, file_name))['train'],
                batch_size,
                device)

        self.n_steps_per_epoch = max(1, len(self.dataset) // batch_size)

        # Accelerate update step
        if device.startswith('cuda'):

            # Warmup and reset gradients
            ref_batches = self.dataset.data[:batch_size*4].to(device).chunk(4)

            for batch in ref_batches[:3]:
                self.optimizer.zero_grad(set_to_none=True)
                self.update(batch)

            # Capture computational graph
            self.optimizer.zero_grad(set_to_none=True)

            self.update, self.graph = capture_graph(
                self.update,
                ref_batches[3].clone(),
                warmup_tensor_list=(),
                single_input=True,
                device=None if device == 'cuda' else device)

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
                if self.mirror_prob and self.ckpter.rng.random() < self.mirror_prob:
                    batch = batch.flip(-1)

                if self.graph is None:
                    self.optimizer.zero_grad()

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

    def update(self, batch: Tensor):

        # Unpack inputs and targets
        ipt, targets, weights = batch.split(cfg.ENC_IMG_CHANNEL_SPLIT, dim=1)

        dep_target = ipt[:, -1:]
        clr_target = targets[:, 1].long()
        typ_target = targets[:, 0].long()
        weight_sum = weights.sum()

        # Unpack model outputs
        out = self.model(ipt)

        clr_logits, typ_logits, dep = out.split(cfg.DEC_IMG_CHANNEL_SPLIT, dim=1)

        # Logits to probs.
        clr_log_probs = log_softmax(clr_logits, dim=1)
        clr_probs = clr_logits.softmax(dim=1)

        typ_log_probs = log_softmax(typ_logits, dim=1)
        typ_probs = typ_logits.softmax(dim=1)

        # Expand indices to probs. (one-hot)
        clr_probs_target = one_hot(clr_target, cfg.N_ALL_CLR_CLASSES)
        clr_probs_target = torch.movedim(clr_probs_target, -1, 1)

        typ_probs_target = one_hot(typ_target, cfg.N_SEG_CLASSES)
        typ_probs_target = torch.movedim(typ_probs_target, -1, 1)

        # NOTE: Using weighted sum / weight_sum instead of mean, weighting across the whole batch
        # Mean squared error
        dep_mse = ((dep - dep_target).square() * weights).sum() / weight_sum

        # Focal cross entropy (gamma 2)
        clr_fce = ((1. - clr_probs).square() * (clr_log_probs * clr_probs_target.neg_()) * weights).sum() / weight_sum

        # Focal cross entropy (gamma 2)
        typ_fce = ((1. - typ_probs).square() * (typ_log_probs * typ_probs_target.neg_()) * weights).sum() / weight_sum

        # Update params.
        loss = dep_mse + clr_fce + typ_fce
        loss.backward()

        self.optimizer.step()

        # Stats for logging
        with torch.no_grad():

            # NOTE: Loss is later divided by iter_step, hence noisy at the start of each epoch
            self.loss += loss.detach()

            self.stats['visnet/loss'] += loss.detach()
            self.stats['visnet/dep_mse'] += dep_mse.detach()
            self.stats['visnet/clr_fce'] += clr_fce.detach()
            self.stats['visnet/typ_fce'] += typ_fce.detach()


class Tester:
    MAX_DISP_SECONDS = Trainer.MAX_DISP_SECONDS

    METRICS = ('clr_fce', 'typ_fce', 'dep_wse', 'loss', 'clr_acc', 'typ_acc', 'dep_err', 'enc_sim')
    TYPE_CLASSES = ('any', 'sky', 'ground', 'wall', 'obj', 'bot', 'body', 'cargo')
    PERCENTILES = ('min', '0.1p', '1p', '10p', '33p', 'median', 'mean', '67p', '90p', '99p', '99.9p', 'max')

    def __init__(self, args: Namespace):

        # Unpack args
        model_prefix: str = args.model_name
        device: str = args.device
        file_name: str = args.file_name

        # Init. model
        self.model = VisNet().to(device)
        self.device = torch.device(device)

        self.model_names = [
            model_name
            for model_name in os.listdir(cfg.DATA_DIR)
            if model_name.startswith(model_prefix)]

        # Logging
        self.loss = torch.tensor(0.)
        self.stat_path = os.path.join(cfg.DATA_DIR, file_name[:-4] + '_test_stat.json')
        self.res_path = os.path.join(cfg.DATA_DIR, file_name[:-4] + '_test_res.json')

        self.stat_keys = [
            f'{metric}/{type_class}/{percentile}'
            for metric in self.METRICS
            for type_class in self.TYPE_CLASSES
            for percentile in self.PERCENTILES]

        self.stats = {
            model_name: {stat_key: [] for stat_key in self.stat_keys}
            for model_name in self.model_names}

        # Dataset
        self.dataset = torch.from_numpy(np.load(os.path.join(cfg.DATA_DIR, file_name))['test']).to(device)

        self.n_epochs = len(self.model_names)
        self.n_steps_per_epoch = None

        # Accelerate eval. step
        if device.startswith('cuda'):

            # Warmup
            for _ in range(3):
                self.get_eval_data()

            # Capture computational graph
            self.get_eval_data, self.graph = capture_graph(
                self.get_eval_data,
                (),
                warmup_tensor_list=(),
                device=None if device == 'cuda' else device)

        else:
            self.graph = None

    def run(self):
        starting_step = 0
        starting_time = perf_counter()

        for epoch_step, model_name in enumerate(self.model_names, start=1):

            # Estimate time remaining
            progress = epoch_step / self.n_epochs
            running_time = perf_counter() - starting_time

            remaining_time = min(
                int(running_time * (self.n_epochs - epoch_step) / (epoch_step - starting_step)),
                self.MAX_DISP_SECONDS)

            # Iterate over model versions
            model_dir = os.path.join(cfg.DATA_DIR, model_name)
            ver_names = [ver_name for ver_name in os.listdir(model_dir) if ver_name.startswith('model')]
            ver_paths = [os.path.join(model_dir, ver_name) for ver_name in ver_names]

            self.n_steps_per_epoch = len(ver_paths)

            for iter_step, (ver_name, ver_path) in enumerate(zip(ver_names, ver_paths), start=1):
                Trainer.print_progress(self, progress, remaining_time, epoch_step, iter_step)

                try:
                    self.model.load_state_dict(torch.load(ver_path, map_location=self.device))

                except RuntimeError:
                    continue

                # Run model
                data = self.get_eval_data()

                # Extract stats
                self.update_stats(data, model_name, ver_name)

            # Save stats
            self.checkpoint()

        # Identify best performer for each category
        self.get_candidates()

    def checkpoint(self):
        with open(self.stat_path, 'w') as statfile:
            json.dump(self.stats, statfile)

    def get_eval_data(self) -> 'tuple[Tensor, ...]':
        with torch.inference_mode():

            # Unpack inputs and targets
            ipt, targets, weights = self.dataset.split(cfg.ENC_IMG_CHANNEL_SPLIT, dim=1)

            dep_target = ipt[:, -1:]
            clr_target = targets[:, 1].long()
            typ_target = targets[:, 0].long()

            # Unpack model outputs
            enc = self.model.enc(ipt)
            out = self.model.dec(enc)

            enc = enc.flatten(1)
            clr_logits, typ_logits, dep = out.split(cfg.DEC_IMG_CHANNEL_SPLIT, dim=1)

            # Logits to probs.
            clr_log_probs = log_softmax(clr_logits, dim=1)
            clr_probs = clr_logits.softmax(dim=1)

            typ_log_probs = log_softmax(typ_logits, dim=1)
            typ_probs = typ_logits.softmax(dim=1)

            # Expand indices to probs. (one-hot)
            clr_probs_target = one_hot(clr_target, cfg.N_ALL_CLR_CLASSES)
            clr_probs_target = torch.movedim(clr_probs_target, -1, 1)

            typ_probs_target = one_hot(typ_target, cfg.N_SEG_CLASSES)
            typ_probs_target = torch.movedim(typ_probs_target, -1, 1)

            # Weighted squared error
            dep_wse = (dep - dep_target).square() * weights

            # Focal cross entropy (gamma 2)
            clr_fce = (1. - clr_probs).square() * (clr_log_probs * clr_probs_target.neg_())
            clr_fce = clr_fce.sum(dim=1, keepdim=True) * weights

            # Focal cross entropy (gamma 2)
            typ_fce = (1. - typ_probs).square() * (typ_log_probs * typ_probs_target.neg_())
            typ_fce = typ_fce.sum(dim=1, keepdim=True) * weights

            loss = dep_wse + clr_fce + typ_fce

            # Accuracy
            clr_acc = (clr_probs.argmax(dim=1) == clr_target).float()
            typ_acc = (typ_probs.argmax(dim=1) == typ_target).float()
            dep_err = (dep - dep_target).abs()

            # Pairwise encoding similarity
            enc_sim = torch.linalg.norm(enc.unsqueeze(1) - enc.unsqueeze(0), dim=-1)

        return clr_fce, typ_fce, dep_wse, loss, clr_acc, typ_acc, dep_err, enc_sim, typ_target

    def update_stats(self, data: 'tuple[Tensor, ...]', model_name: str, ver_name: str):
        *arrays, type_target = [tensor.flatten().cpu().numpy().astype(np.float64) for tensor in data]

        for metric, array in zip(self.METRICS, arrays):

            # Special case (values only 0 or 1)
            if 'acc' in metric:
                stat = array.mean()

                self.stats[model_name][f'{metric}/{self.TYPE_CLASSES[0]}/mean'].append((ver_name, stat))

                for type_id, type_class in enumerate(self.TYPE_CLASSES[1:]):
                    stat = array[type_target == type_id].mean()

                    self.stats[model_name][f'{metric}/{type_class}/mean'].append((ver_name, stat))

                continue

            # Over all
            stats = self.get_stats(array)

            for percentile, stat in zip(self.PERCENTILES, stats):
                self.stats[model_name][f'{metric}/{self.TYPE_CLASSES[0]}/{percentile}'].append((ver_name, stat))

            # Per class
            if metric == 'enc_sim':
                continue

            for type_id, type_class in enumerate(self.TYPE_CLASSES[1:]):
                stats = self.get_stats(array[type_target == type_id])

                for percentile, stat in zip(self.PERCENTILES, stats):
                    self.stats[model_name][f'{metric}/{type_class}/{percentile}'].append((ver_name, stat))

    @staticmethod
    def get_stats(arr: np.ndarray) -> 'tuple[float, ...]':
        return (
            np.min(arr),
            np.percentile(arr, 0.1),
            np.percentile(arr, 1.),
            np.percentile(arr, 10.),
            np.percentile(arr, 33.),
            np.median(arr),
            np.mean(arr),
            np.percentile(arr, 67.),
            np.percentile(arr, 90.),
            np.percentile(arr, 99.),
            np.percentile(arr, 99.9),
            np.max(arr))

    def get_candidates(self):
        results = {}
        candidate_scores = {}

        for stat_key in self.stat_keys:
            best_model = None

            # Prefer higher score
            if 'acc' in stat_key or 'enc_sim' in stat_key:
                best_score = 0.

                def cmp_fn(score: float, best_score: float) -> bool:
                    return score > best_score

            # Prefer lower score
            else:
                best_score = np.inf

                def cmp_fn(score: float, best_score: float) -> bool:
                    return score < best_score

            # Iterate over models
            for model_name, model_stats in self.stats.items():
                for ver_name, score in model_stats[stat_key]:
                    if cmp_fn(score, best_score):
                        best_score = score
                        best_model = (model_name, ver_name)

            # Update candidate scores
            if best_model is not None:
                results[stat_key] = (*best_model, best_score)

                model_key = f'{best_model[0]}/{best_model[1]}'

                if model_key in candidate_scores:
                    candidate_scores[model_key] += 1

                else:
                    candidate_scores[model_key] = 1

        results['candidate_scores'] = candidate_scores

        # Save results
        with open(self.res_path, 'w') as resfile:
            json.dump(results, resfile)

        candidates = sorted(candidate_scores.keys())

        print('\n\nCandidate_scores:')

        for candidate in candidates:
            print(f'{candidate}: {candidate_scores[candidate]}')


class Teachers:
    MAX_DISP_SECONDS = Trainer.MAX_DISP_SECONDS

    def __init__(self, args: Namespace):

        # Unpack args
        model_name: str = args.model_name
        device: str = args.device
        file_name, subset_key = args.file_name.split(':')
        sideload: bool = args.sideload
        self.batch_size: int = args.batch_size

        # Init. models
        ver_names = [
            ver_name
            for ver_name in os.listdir(os.path.join(cfg.DATA_DIR, model_name))
            if ver_name.startswith('model')]

        self.models = []
        self.ver_names = []

        for ver_name in ver_names:
            model = VisNet().to(device)

            try:
                model.load_state_dict(torch.load(os.path.join(cfg.DATA_DIR, model_name, ver_name), map_location=device))

            except RuntimeError:
                pass

            self.models.append(model)
            self.ver_names.append(ver_name)

        # Dataset
        self.dataset = (
            SideLoadingDataset
            if device.startswith('cuda') and sideload
            else LoadedDataset)(
                np.load(os.path.join(cfg.DATA_DIR, file_name))[subset_key],
                self.batch_size,
                device,
                shuffle=False)

        self.n_epochs = 1
        self.n_steps_per_epoch = max(1, len(self.dataset) // self.batch_size)

        self.proc_set = np.empty(
            (
                self.n_steps_per_epoch * self.batch_size,
                sum(cfg.ENC_IMG_CHANNEL_SPLIT) + sum(cfg.DEC_IMG_CHANNEL_SPLIT),
                cfg.OBS_IMG_RES_HEIGHT,
                cfg.OBS_IMG_RES_WIDTH),
            dtype=np.float32)

        self.proc_path = os.path.join(cfg.DATA_DIR, file_name[:-4] + '_proc.npz')
        self.stud_path = os.path.join(cfg.DATA_DIR, file_name[:-4] + '_stud.pt')

        # Logging
        self.loss = torch.tensor(0.)

        # Accelerate proc. step
        if device.startswith('cuda'):

            # Warmup
            ref_batches = self.dataset.data[:self.batch_size*4].to(device).chunk(4)

            # Capture computational graph
            self.update, self.graph = capture_graph(
                self.proc_data,
                ref_batches[3].clone(),
                warmup_tensor_list=ref_batches[:3],
                single_input=True,
                device=None if device == 'cuda' else device)

        else:
            self.graph = None

    def run(self):
        starting_step = 0
        starting_time = perf_counter()

        for epoch_step in range(starting_step+1, self.n_epochs+1):

            # Estimate time remaining
            progress = epoch_step / self.n_epochs
            running_time = perf_counter() - starting_time

            remaining_time = min(
                int(running_time * (self.n_epochs - epoch_step) / (epoch_step - starting_step)),
                self.MAX_DISP_SECONDS)

            for iter_step, batch in enumerate(self.dataset, start=1):
                Trainer.print_progress(self, progress, remaining_time, epoch_step, iter_step)

                out = self.update(batch)
                self.proc_set[(iter_step-1)*self.batch_size:iter_step*self.batch_size] = out.cpu().numpy()

        self.checkpoint()

    def checkpoint(self):
        np.savez_compressed(self.proc_path, data=self.proc_set)

        state_dicts = [model.state_dict() for model in self.models]
        student_state_dict = {k: v for k, v in state_dicts[0].items()}

        # Average parameters
        for key in student_state_dict:
            for state_dict in state_dicts[1:]:
                student_state_dict[key] = student_state_dict[key] + state_dict[key]

            student_state_dict[key] = student_state_dict[key] / len(state_dicts)

        torch.save(student_state_dict, self.stud_path)

    def proc_data(self, batch: Tensor) -> Tensor:
        proc_out = losses = None

        with torch.inference_mode():

            # Unpack inputs and targets
            ipt, targets, weights = batch.split(cfg.ENC_IMG_CHANNEL_SPLIT, dim=1)

            dep_target = ipt[:, -1:]
            clr_target = targets[:, 1].long()
            typ_target = targets[:, 0].long()

            # Expand indices to probs. (one-hot)
            clr_probs_target = one_hot(clr_target, cfg.N_ALL_CLR_CLASSES).neg_()
            clr_probs_target = torch.movedim(clr_probs_target, -1, 1)

            typ_probs_target = one_hot(typ_target, cfg.N_SEG_CLASSES).neg_()
            typ_probs_target = torch.movedim(typ_probs_target, -1, 1)

            for model in self.models:

                # Unpack model outputs
                out = model(ipt)

                clr_logits, typ_logits, dep = out.split(cfg.DEC_IMG_CHANNEL_SPLIT, dim=1)

                # Logits to probs.
                clr_log_probs = log_softmax(clr_logits, dim=1)
                typ_log_probs = log_softmax(typ_logits, dim=1)

                # Absolute error
                dep_err = (dep - dep_target).abs()

                # Cross entropy
                clr_ce = clr_log_probs * clr_probs_target

                # Cross entropy
                typ_ce = typ_log_probs * typ_probs_target

                if proc_out is None:
                    proc_out = clr_logits, typ_logits, dep
                    losses = clr_ce, typ_ce, dep_err

                    continue

                clr_logits_prev, typ_logits_prev, dep_prev = proc_out
                clr_ce_prev, typ_ce_prev, dep_err_prev = losses

                clr_mask = (clr_ce < clr_ce_prev).float()
                typ_mask = (typ_ce < typ_ce_prev).float()
                dep_mask = (dep_err < dep_err_prev).float()

                proc_out = (
                    torch.lerp(clr_logits_prev, clr_logits, clr_mask),
                    torch.lerp(typ_logits_prev, typ_logits, typ_mask),
                    torch.lerp(dep_prev, dep, dep_mask))

                losses = (
                    torch.lerp(clr_ce_prev, clr_ce, clr_mask),
                    torch.lerp(typ_ce_prev, typ_ce, typ_mask),
                    torch.lerp(dep_err_prev, dep_err, dep_mask))

            clr_logits, typ_logits, dep = proc_out
            clr_probs = clr_logits.softmax(dim=1)
            typ_probs = typ_logits.softmax(dim=1)

            out = torch.cat((batch, clr_probs, typ_probs, dep), dim=1)

        return out


class Student:
    MAX_DISP_SECONDS = Trainer.MAX_DISP_SECONDS

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

        # Init. model
        if device.startswith('cuda'):
            torch.backends.cudnn.benchmark = True

        self.ckpter = CheckpointTracker(model_name, cfg.DATA_DIR, device, rng_seed)

        self.model = VisNet()
        self.optimizer = NAdamW(self.model.parameters(), lr=args.lr_max, weight_decay=1e-2, device=device)

        self.model.load_state_dict(
            torch.load(os.path.join(cfg.DATA_DIR, file_name[:-13] + '_stud.pt'), map_location=device))

        self.ckpter.load_model(self.model, self.optimizer)

        self.scheduler = AnnealingScheduler(
            self.optimizer,
            step_milestones=[lr_main_start_epoch, lr_main_end_epoch, self.n_epochs],
            lr_div_factors=[args.lr_max/args.lr_init, args.lr_max/args.lr_final],
            starting_step=self.ckpter.meta['update_step'])

        # Logging
        self.loss = torch.tensor(0., device=device)
        new_zero_tensor = self.loss.clone

        self.stats = {
            'visnet/dep_mse': new_zero_tensor(),
            'visnet/clr_fce': new_zero_tensor(),
            'visnet/typ_fce': new_zero_tensor(),
            'distil/loss': new_zero_tensor(),
            'distil/dep_mse': new_zero_tensor(),
            'distil/clr_fce': new_zero_tensor(),
            'distil/typ_fce': new_zero_tensor()}

        self.writer = SummaryWriter(log_dir=os.path.join(cfg.LOG_DIR, model_name))
        self.write = self.writer.add_scalar

        # Dataset
        self.dataset = (
            SideLoadingDataset
            if device.startswith('cuda') and sideload
            else LoadedDataset)(
                np.load(os.path.join(cfg.DATA_DIR, file_name))['data'],
                batch_size,
                device)

        self.n_steps_per_epoch = max(1, len(self.dataset) // batch_size)

        # Accelerate update step
        if device.startswith('cuda'):

            # Warmup and reset gradients
            ref_batches = self.dataset.data[:batch_size*4].to(device).chunk(4)

            for batch in ref_batches[:3]:
                self.optimizer.zero_grad(set_to_none=True)
                self.update(batch)

            # Capture computational graph
            self.optimizer.zero_grad(set_to_none=True)

            self.update, self.graph = capture_graph(
                self.update,
                ref_batches[3].clone(),
                warmup_tensor_list=(),
                single_input=True,
                device=None if device == 'cuda' else device)

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

                if self.graph is None:
                    self.optimizer.zero_grad()

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

    def update(self, batch: Tensor):

        # Unpack inputs and targets
        ipt, targets, weights, clr_probs_t, typ_probs_t, dep_t = batch.split(cfg.ALL_IMG_CHANNEL_SPLIT, dim=1)

        dep_target = ipt[:, -1:]
        clr_target = targets[:, 1].long()
        typ_target = targets[:, 0].long()
        weight_sum = weights.sum()

        # Unpack model outputs
        out = self.model(ipt)

        clr_logits, typ_logits, dep = out.split(cfg.DEC_IMG_CHANNEL_SPLIT, dim=1)

        # Logits to probs.
        clr_log_probs = log_softmax(clr_logits, dim=1)
        clr_probs = clr_logits.softmax(dim=1)

        typ_log_probs = log_softmax(typ_logits, dim=1)
        typ_probs = typ_logits.softmax(dim=1)

        # Expand indices to probs. (one-hot)
        clr_probs_target = one_hot(clr_target, cfg.N_ALL_CLR_CLASSES)
        clr_probs_target = torch.movedim(clr_probs_target, -1, 1)

        typ_probs_target = one_hot(typ_target, cfg.N_SEG_CLASSES)
        typ_probs_target = torch.movedim(typ_probs_target, -1, 1)

        # NOTE: Using weighted sum / weight_sum instead of mean, weighting across the whole batch
        # Mean squared error
        dep_mse = ((dep - dep_target).square() * weights).sum() / weight_sum
        dep_mse_t = ((dep - dep_t).square() * weights).sum() / weight_sum

        # Focal cross entropy (gamma 2)
        clr_fce = ((1. - clr_probs).square() * (clr_log_probs * clr_probs_target.neg_()) * weights).sum() / weight_sum
        clr_fce_t = ((1. - clr_probs).square() * (clr_log_probs * clr_probs_t.neg()) * weights).sum() / weight_sum

        # Focal cross entropy (gamma 2)
        typ_fce = ((1. - typ_probs).square() * (typ_log_probs * typ_probs_target.neg_()) * weights).sum() / weight_sum
        typ_fce_t = ((1. - typ_probs).square() * (typ_log_probs * typ_probs_t.neg()) * weights).sum() / weight_sum

        # Update params.
        loss = dep_mse + clr_fce + typ_fce + dep_mse_t + clr_fce_t + typ_fce_t
        loss.backward()

        self.optimizer.step()

        # Stats for logging
        with torch.no_grad():

            # NOTE: Loss is later divided by iter_step, hence noisy at the start of each epoch
            self.loss += loss.detach()

            self.stats['visnet/dep_mse'] += dep_mse.detach()
            self.stats['visnet/clr_fce'] += clr_fce.detach()
            self.stats['visnet/typ_fce'] += typ_fce.detach()
            self.stats['distil/loss'] += loss.detach()
            self.stats['distil/dep_mse'] += dep_mse_t.detach()
            self.stats['distil/clr_fce'] += clr_fce_t.detach()
            self.stats['distil/typ_fce'] += typ_fce_t.detach()


if __name__ == '__main__':
    parser = ArgumentParser(description='Vis. enc. pre-training.')

    parser.add_argument(
        '--mode', type=str, default='trainer',
        help='Process to run (trainer, tester, teachers, student).')

    parser.add_argument(
        '--model_name', type=str, default='visnet',
        help='Name under which model checkpoints and events will be saved.')
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='Device to train on (CPU, GPU, or GPU with index).')
    parser.add_argument(
        '--rng_seed', type=int, default=42,
        help='Seed for initialising random number generators.')

    parser.add_argument(
        '--n_epochs', type=int, default=3000,
        help='Number of repeated iterations through the dataset.')
    parser.add_argument(
        '--log_epoch_interval', type=int, default=1,
        help='Interval for logging current loss components.')
    parser.add_argument(
        '--ckpt_epoch_interval', type=int, default=10,
        help='Interval for saving current model parameters.')
    parser.add_argument(
        '--branch_epoch_interval', type=int, default=100,
        help='Interval for starting a new branch, i.e. path to save current model parameters.')

    parser.add_argument(
        '--lr_main_start_epoch', type=float, default=10,
        help='Epoch on which the learning rate assumes its peak value.')
    parser.add_argument(
        '--lr_main_end_epoch', type=float, default=650,
        help='Epoch on which the learning rate proceeds to cool down.')
    parser.add_argument(
        '--lr_init', type=float, default=5e-3,
        help='Initial learning rate in the warmup phase.')
    parser.add_argument(
        '--lr_max', type=float, default=6e-4,
        help='Maximum learning rate in the main phase.')
    parser.add_argument(
        '--lr_final', type=float, default=1e-7,
        help='Final learning rate in the cooldown phase.')

    parser.add_argument(
        '--file_name', type=str, default='rec_00-06_set.npz',
        help='Name of the file with pre-processed image data.')
    parser.add_argument(
        '--sideload', type=int, default=0,
        help='Option to stream batches from RAM instead of moving the whole dataset to target device.')
    parser.add_argument(
        '--batch_size', type=int, default=128,
        help='Number of images processed simultaneously per update.')
    parser.add_argument(
        '--mirror_prob', type=float, default=0.,
        help='Probability of flipping images in a batch over the horizontal dimension.')

    args = parser.parse_args()
    mode = args.mode.lower()

    if mode == 'trainer':
        runner = Trainer(args)

    elif mode == 'tester':
        runner = Tester(args)

    elif mode == 'teachers':
        runner = Teachers(args)

    elif mode == 'student':
        runner = Student(args)

    else:
        raise ValueError(f'Mode {mode} not recognised.')

    try:
        runner.run()

    except KeyboardInterrupt:
        print('\nProcess interrupted by user.', end='')

    print('\nDone.')
