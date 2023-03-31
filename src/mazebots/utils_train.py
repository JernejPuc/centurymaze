"""Training utilities"""

import json
import gc
import os
from datetime import datetime
from math import cos
from typing import Any

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch import cuda, Tensor
from torch.nn import Module
from torch.optim import Optimizer


class NAdamW(Optimizer):
    """Adam with Nesterov momentum and decoupled weight decay."""

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: 'tuple[float, float, float]' = (0.9, 0.9, 0.98),
        beta_products: 'tuple[float, float]' = (1., 1.),
        eps: float = 1e-6,
        weight_decay: float = 1e-2,
        device: 'str | torch.device' = 'cuda'
    ):
        defaults = dict(
            lr=torch.tensor(lr, dtype=torch.float32, device=device),
            beta1=torch.tensor(betas[0], dtype=torch.float32, device=device),
            beta1_next=torch.tensor(betas[1], dtype=torch.float32, device=device),
            beta2=torch.tensor(betas[2], dtype=torch.float32, device=device),
            beta1_product=torch.tensor(beta_products[0], dtype=torch.float32, device=device),
            beta2_product=torch.tensor(beta_products[1], dtype=torch.float32, device=device),
            eps=torch.tensor(eps, dtype=torch.float32, device=device),
            weight_decay=(torch.tensor(weight_decay, dtype=torch.float32, device=device) if weight_decay else None),
            step=torch.tensor(0, dtype=torch.int64, device=device))

        super().__init__(params, defaults)

        # Init EWMA of gradients and squared gradients
        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    if not p.requires_grad:
                        continue

                    state: 'dict[str, Tensor]' = self.state[p]

                    if len(state) == 0:
                        state['exp_avg'] = torch.zeros_like(p, device=device, memory_format=torch.preserve_format)
                        state['exp_avg_sq'] = torch.zeros_like(p, device=device, memory_format=torch.preserve_format)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:

            # Unpack
            lr = group['lr']
            beta1 = group['beta1']
            beta1_next = group['beta1_next']
            beta2 = group['beta2']
            beta1_product = group['beta1_product']
            beta2_product = group['beta2_product']
            eps = group['eps']
            weight_mul = (1. - lr * group['weight_decay']) if group['weight_decay'] is not None else None

            # Update
            group['step'] += 1
            beta2_product *= beta2
            beta1_product *= beta1
            beta1_product_next = beta1_product * beta1_next

            one_minus_beta1 = 1. - beta1
            one_minus_beta2 = 1. - beta2

            bias_correction2 = 1. - beta2_product
            bias_correction1 = 1. - beta1_product
            bias_correction1_next = 1. - beta1_product_next

            grad_step = -lr * one_minus_beta1 / bias_correction1
            momentum_step = -lr * beta1_next / bias_correction1_next

            # Apply per param with valid grad
            for p in group['params']:
                if p.grad is None:
                    continue

                elif p.grad.is_sparse:
                    raise RuntimeError('Sparse gradients are not supported.')

                # Unpack EWMA of gradients and squared gradients
                state: 'dict[str, Tensor]' = self.state[p]

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                param: Tensor = p
                grad: Tensor = p.grad

                # Decoupled weight decay
                if weight_mul is not None:
                    param.mul_(weight_mul)

                # Update EWMA of gradients and squared gradients
                exp_avg.mul_(beta1).add_(grad * one_minus_beta1)                # alpha
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad * one_minus_beta2)   # value

                denom = exp_avg_sq.div(bias_correction2).sqrt().add_(eps)

                # Update param
                param.addcdiv_(grad * grad_step, denom)                         # value
                param.addcdiv_(exp_avg * momentum_step, denom)                  # value


class SoftConstLRScheduler:
    """
    Adds cosine warmup and cooldown to a constant learning rate schedule,
    resembling a one-cycle scheduler with extended middle elevation (plateau)
    to prolong learning at the maximum rate.

    NOTE: Milestones refer to lr, beta1, and duration of the starting, main,
    and final phase.
    """

    step_ctr: int
    in_main: bool

    def __init__(
        self,
        optimiser: Optimizer,
        step_milestones: 'tuple[int, int, int]',
        starting_step: int = 0,
        lr_milestones: 'tuple[float, float, float]' = (2e-5, 4e-4, 1e-6),
        beta1_milestones: 'tuple[float, float, float]' = (0.9, 0.85, 0.98)
    ):
        self.optimiser = optimiser

        self.lr_init, self.lr_main, self.lr_final = lr_milestones
        self.beta1_init, self.beta1_main, self.beta1_final = beta1_milestones

        self.step_start_main = step_milestones[0]
        self.step_end_main = step_milestones[1] + self.step_start_main
        self.step_total = step_milestones[2] + self.step_end_main

        self.reset(starting_step)

    def reset(self, starting_step: int = 0):
        self.step_ctr = starting_step
        self.in_main = False

    def update_params(self, lr: float, beta1: float):
        with torch.no_grad():
            for param_group in self.optimiser.param_groups:
                param_group['lr'].fill_(lr)
                param_group['beta1'].fill_(beta1)
                param_group['beta1_next'].fill_(beta1)

    def anneal(self, start: float, end: float, phase_ratio: float) -> float:
        """Cosine anneal from start to end as phase_ratio goes from 0 to 1."""

        return end + (start - end) / 2. * (cos(np.pi * phase_ratio) + 1.)

    def step(self, increment: int = 1):
        # Keep constant in main phase
        if self.in_main:
            if self.step_ctr < self.step_end_main:
                self.step_ctr += increment
                return

            else:
                self.in_main = False

        # Get annealed lr and momentum
        if self.step_ctr < self.step_start_main:
            phase_ratio = max(0., min(1., self.step_ctr / self.step_start_main))
            lr = self.anneal(self.lr_init, self.lr_main, phase_ratio)
            beta1 = self.anneal(self.beta1_init, self.beta1_main, phase_ratio)

        elif self.step_ctr >= self.step_end_main:
            phase_ratio = max(0., min(1., (self.step_ctr-self.step_end_main) / (self.step_total-self.step_end_main)))
            lr = self.anneal(self.lr_main, self.lr_final, phase_ratio)
            beta1 = self.anneal(self.beta1_main, self.beta1_final, phase_ratio)

        else:
            self.in_main = True
            lr = self.lr_main
            beta1 = self.beta1_main

        # Update
        self.update_params(lr, beta1)
        self.step_ctr += increment


class LoadedDataset:
    """Iterator slicing through data that is fully loaded on the target device."""

    def __init__(
        self,
        file_path: str,
        batch_size: int,
        device: str = 'cuda',
        shuffle: bool = True,
        shuffle_on_cpu: bool = False
    ):
        self.data = torch.load(file_path).to(device)
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.shuffle = shuffle
        self.shuffle_on_cpu = shuffle_on_cpu
        self.iter_ptr = self.len = len(self.data)

    def __len__(self) -> int:
        return self.len

    def __iter__(self) -> 'LoadedDataset':
        if self.shuffle:
            if self.shuffle_on_cpu:
                self.data = self.data.cpu()
                gc.collect()

                self.data = self.data[torch.randperm(self.len)].to(self.device)

            else:
                self.data = self.data[torch.randperm(self.len, device=self.device)]

        self.iter_ptr = -self.batch_size

        return self

    def __next__(self) -> Tensor:
        self.iter_ptr += self.batch_size

        if self.iter_ptr == self.len:
            raise StopIteration

        return self.data[self.iter_ptr:self.iter_ptr+self.batch_size]


class SideLoadingDataset:
    """
    Iterator sideloading batches of data from regular RAM to target CUDA device.

    NOTE: Data is not streamed from storage, so RAM itself must still be able
    to hold at least the equivalent of two datasets, as shuffling and pinning
    memory both make a copy and memory might also not be released instantly.

    Reference:
    https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py#L265
    """

    def __init__(
        self,
        file_path: str,
        batch_size: int,
        device: str = 'cuda',
        shuffle: bool = True,
        pin_memory: bool = True
    ):
        assert device.startswith('cuda'), f'Dataset expected cuda, got {device}.'

        self.data = torch.load(file_path)
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.iter_ptr = self.len = len(self.data)

        if pin_memory:
            self.data = self.data.pin_memory()

        # Side stream
        self.stream = cuda.Stream()
        self.next_batch = None

    def __len__(self) -> int:
        return self.len

    def __iter__(self) -> 'SideLoadingDataset':
        if self.shuffle:
            self.data = self.data[torch.randperm(self.len)]

            if self.pin_memory:
                gc.collect()

                self.data = self.data.pin_memory()

        self.iter_ptr = -self.batch_size

        # Load on side stream
        with cuda.stream(self.stream):
            self.next_batch = self.data[-self.batch_size:].to(self.device, non_blocking=True)

        return self

    def __next__(self) -> Tensor:
        self.iter_ptr += self.batch_size

        if self.iter_ptr == self.len:
            self.next_batch = None

            raise StopIteration

        # Wait until loading in side stream completes
        cuda.current_stream().wait_stream(self.stream)

        batch = self.next_batch

        # Signal side stream not to reuse this memory while the main stream can work on it
        batch.record_stream(cuda.current_stream())

        # Load on side stream
        with cuda.stream(self.stream):
            self.next_batch = self.data[self.iter_ptr:self.iter_ptr+self.batch_size].to(self.device, non_blocking=True)

        return batch


class Batch(dict):
    """Wrapper around a dict with assumptions for key and value content."""

    VAL_TYPES = {
        'act': 'tuple[Tensor, ...]',
        'val': Tensor,
        'obs': 'tuple[Tensor, ...]',
        'mem': 'tuple[Tensor, ...]',
        'rew': Tensor,
        'ret': Tensor,
        'adv': Tensor,
        'rst': Tensor,
        'nrst': Tensor}

    def size(self) -> int:
        try:
            size = next(len(v[0] if isinstance(v, tuple) else v) for v in self.values())

        except StopIteration:
            raise RuntimeError('Tried to infer batch size from empty batch.')

        return size

    def device(self) -> torch.device:
        try:
            device = next((v[0] if isinstance(v, tuple) else v).device for v in self.values())

        except StopIteration:
            raise RuntimeError('Tried to infer device from empty batch.')

        return device

    def to_list(self) -> 'list[Tensor]':
        return [
            t
            for v in self.values()
            for t in (v if isinstance(v, tuple) else (v,))]

    def from_list(self, lst: 'list[Tensor]') -> 'Batch':
        lst = iter(lst)

        return Batch({
            k: (
                tuple([next(lst) for _ in range(len(v))])
                if isinstance(v, tuple)
                else next(lst))
            for k, v in self.items()})

    def slice(self, arg: 'int | slice | Ellipsis | ArrayLike') -> 'Batch':
        return Batch({
            k: (
                tuple([t[arg] for t in v])
                if isinstance(v, tuple)
                else v[arg])
            for k, v in self.items()})

    def cat_alike(self, batches: 'list[Batch]') -> 'Batch':
        return Batch({
            k: (
                tuple([torch.cat([b[k][i] for b in batches]) for i in range(len(v))])
                if isinstance(v, tuple)
                else torch.cat([b[k] for b in batches]))
            for k, v in self.items()})


class ExperienceBuffer:
    """
    Wrapper around a list with fixed length, storing state transitions
    (obs., act., rew., etc.) to form trajectories for eventual optimisation.

    NOTE: The implementation lacks checks to handle e.g. tensor content
    of different sizes or changes to the buffer mid-iteration.
    """

    batches: 'list[Batch | None]'
    bind_ptr: int

    def __init__(self, buffer_len: int, data: 'tuple[list[Batch | None], int]' = None):
        if buffer_len < 1:
            raise ValueError(f'Invalid buffer length: {buffer_len}')

        self.buffer_len = buffer_len
        self.iter_step = 1
        self.iter_ptr = buffer_len
        self.item_iter = False

        if data is None:
            self.clear()

        else:
            self.batches, self.bind_ptr = data

    def clear(self):
        self.batches = [None] * self.buffer_len
        self.bind_ptr = 0

    def is_full(self) -> bool:
        return self.bind_ptr == self.buffer_len

    def size(self) -> int:
        return self.buffer_len

    def batch_size(self) -> int:
        try:
            return self.batches[0].size()

        except AttributeError as err:
            raise RuntimeError('Tried to infer batch size from empty buffer.') from err

    def device(self) -> torch.device:
        try:
            return self.batches[0].device()

        except AttributeError as err:
            raise RuntimeError('Tried to infer device from empty buffer.') from err

    def append(self, val: Batch):
        if self.bind_ptr == self.buffer_len:
            raise IndexError('Appended to full buffer.')

        self.batches[self.bind_ptr] = val
        self.bind_ptr += 1

    def pop(self) -> Batch:
        if self.bind_ptr == 0:
            raise IndexError('Popped from empty buffer.')

        val = self.batches[self.bind_ptr]

        self.batches[self.bind_ptr] = None
        self.bind_ptr -= 1

        return val

    def __setitem__(self, *args):
        raise RuntimeError('Arbitrary assignment is prohibited. Use append & pop instead.')

    def __getitem__(self, arg: 'int | slice | tuple[int | slice, int | slice]') -> 'Batch | ExperienceBuffer':
        if isinstance(arg, int):
            return self.batches[arg]

        if isinstance(arg, slice):
            if arg.start is arg.stop is arg.step is None:
                return self

            batches = self.batches[arg]
            bind_ptr = min(self.bind_ptr, len(batches))

            return ExperienceBuffer(len(batches), data=(batches, bind_ptr))

        if isinstance(arg, tuple):
            buffer_arg, batch_arg = arg

            if isinstance(buffer_arg, int):
                batch = self[buffer_arg]

                return None if batch is None else batch.slice(batch_arg)

            buffer = self[buffer_arg]
            buffer.batches = [None if b is None else b.slice(batch_arg) for b in buffer.batches]

            return buffer

        else:
            raise TypeError(f'Invalid indexing argument: {arg}')

    def __len__(self) -> int:
        return self.bind_ptr

    def __iter__(self) -> 'ExperienceBuffer':
        if self.bind_ptr < self.buffer_len:
            raise RuntimeError('Need full buffer to iterate.')

        self.iter_ptr = -self.iter_step

        return self

    def __next__(self) -> 'list[Batch] | Batch':
        self.iter_ptr += self.iter_step

        if self.iter_ptr == self.buffer_len:
            self.iter_step = 1
            self.item_iter = False

            raise StopIteration

        # Return a single batch
        if self.item_iter:
            return self.batches[self.iter_ptr]

        # Return a sublist of batches
        return self.batches[self.iter_ptr:self.iter_ptr+self.iter_step]

    def iter_slices(self, iter_step: int):
        if iter_step < 1:
            raise ValueError(f'Invalid iter. step: {iter_step}')

        if self.buffer_len % iter_step:
            raise ValueError(f'Iter. step ({iter_step}) inconsistent with buffer length ({self.buffer_len}).')

        self.iter_step = iter_step

        return iter(self)

    def iter_items(self):
        self.item_iter = True

        return iter(self)

    def shuffle(
        self,
        rng: np.random.Generator,
        key: str = None,
        batch_wise: bool = False,
        alpha: float = 0.7,
        beta: float = 0.5,
        eps: float = 0.
    ) -> 'None | Tensor | list[Tensor]':

        if self.bind_ptr < self.buffer_len:
            raise RuntimeError('Need full buffer to shuffle.')

        if batch_wise:
            batch_size = self.batch_size()

            # Reorder samples per batch wrt. permutation (uniform probs.)
            if key is None:
                probs = None
                idcs = [rng.permutation(batch_size) for _ in self.batches]

            # Probs. wrt. key, e.g. advantages
            # NOTE: Using abs, assuming advantages
            else:
                try:
                    with torch.no_grad():
                        probs = [b[key].abs().cpu().numpy() ** alpha + eps for b in self.batches]
                        probs = [prob / prob.sum() for prob in probs]

                except KeyError as err:
                    raise RuntimeError('Tried to prio. shuffle with unlabelled samples.') from err

                # Sample index sequences
                idcs = [rng.choice(batch_size, batch_size, replace=False, p=p) for p in probs]

            # Reorder samples per batch
            self.batches = [b.slice(i) for b, i in zip(self.batches, idcs)]

            if probs is None:
                return

            # Weights to scale loss contributions
            weights = [(p[i] * batch_size) ** -beta for p, i in zip(probs, idcs)]
            weights = [
                torch.from_numpy(w / w.max()).to(self.batches[0][key].device, dtype=torch.float32)
                for w in weights]

            return weights

        # Reorder batches wrt. permutation (uniform probs.)
        if key is None:
            self.batches = [self.batches[i] for i in rng.permutation(self.buffer_len)]

            return

        # Probs. wrt. key, e.g. advantages
        # NOTE: Using abs, assuming advantages
        try:
            with torch.no_grad():
                probs = np.array([b[key].abs().mean().item() for b in self.batches]) ** alpha + eps
                probs /= probs.sum()

        except KeyError as err:
            raise RuntimeError('Tried to prio. shuffle with unlabelled samples.') from err

        # Sample index sequence
        idcs = rng.choice(self.buffer_len, self.buffer_len, replace=False, p=probs)

        # Reorder batches
        self.batches = [self.batches[i] for i in idcs]

        # Weights to scale loss contributions
        weights = (probs[idcs] * self.buffer_len) ** -beta
        weights = torch.from_numpy(weights / weights.max()).to(self.batches[0][key].device, dtype=torch.float32)

        return weights

    def stack(self, n_slices: int) -> 'ExperienceBuffer':
        """Increase batch size by stacking multiple buffer slices."""

        if n_slices < 2:
            raise ValueError(f'Invalid slice num.: {n_slices}')

        if self.buffer_len % n_slices or n_slices > self.buffer_len:
            raise ValueError(f'Slice num. ({n_slices}) inconsistent with buffer length ({self.buffer_len}).')

        if self.bind_ptr < self.buffer_len:
            raise RuntimeError('Need full buffer to stack.')

        slice_len = self.buffer_len // n_slices
        batch_ref = self.batches[0]

        batches = [batch_ref.cat_alike(self.batches[i::slice_len]) for i in range(slice_len)]

        return ExperienceBuffer(slice_len, data=(batches, slice_len))

    def unstack(self, n_slices: int) -> 'ExperienceBuffer':
        """Reduce batch size by splitting batches into multiple buffer slices."""

        if n_slices < 2:
            raise ValueError(f'Invalid slice num.: {n_slices}')

        if self.bind_ptr < self.buffer_len:
            raise RuntimeError('Need full buffer to unstack.')

        batch_size = self.batch_size()

        if batch_size % n_slices or n_slices > batch_size:
            raise ValueError(f'Slice num. ({n_slices}) inconsistent with batch size ({batch_size}).')

        buffer_len = self.buffer_len * n_slices
        new_size = batch_size // n_slices
        slices = [slice(i, i+new_size) for i in range(0, batch_size, new_size)]

        batches = [b.slice(s) for s in slices for b in self.batches]

        return ExperienceBuffer(buffer_len, data=(batches, buffer_len))

    def label(self, values: Tensor, gamma: float, lam: float) -> 'tuple[float, float]':
        """
        Compute advantages and returns via generalised advantage estimation (GAE)
        in a traceable way and add them to existing batches.
        """

        if not all(self.batches):
            raise RuntimeError('Need full buffer to label.')

        advantages = torch.zeros_like(self.batches[-1]['rew'])

        # NOTE: Non-reset (non-terminal) mask zeroes out values between resets
        for batch in reversed(self.batches):
            deltas = batch['rew'] + batch['nrst'] * gamma * values - batch['val']
            advantages = deltas + batch['nrst'] * gamma * lam * advantages
            values = batch['val']

            batch['adv'] = advantages
            batch['ret'] = advantages + values

        advantages = torch.stack([batch['adv'] for batch in self.batches], dim=1)

        # NOTE: Standardisation works best over the whole rollout (more samples, fewer outliers)
        # NOTE: Div. by scale is clipped to limit the noise of sparse rewards (max. 10x larger)
        adv_mean = advantages.mean()
        adv_std = advantages.std()

        advantages = (advantages - adv_mean) / torch.clip(adv_std, 0.1)

        for i, batch in enumerate(self.batches):
            batch['adv'] = advantages[:, i]

        return adv_mean.item(), adv_std.item()


class CheckpointTracker:
    meta: 'dict[str, Any]'
    rng: np.random.Generator

    def __init__(
        self,
        model_name: str = 'model',
        data_dir: str = 'data',
        device: str = 'cuda',
        initial_seed: int = 42,
        deterministic: bool = False
    ):
        if deterministic:
            # See: https://github.com/pytorch/pytorch/issues/76176
            print('Warning: Some algorithms have unresolved issues in deterministic mode.')

            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)

        self.model_name = model_name
        self.data_dir = os.path.join(data_dir, model_name)
        self.device = device

        self.resume(initial_seed)

        self.model: 'Module | None' = None
        self.optimiser: 'Optimizer | None' = None

    def resume(self, seed: int):
        """Initialise the first or load the last checkpoint data of the current model."""

        # Load or create meta.json file
        if os.path.exists(path := os.path.join(self.data_dir, 'meta.json')):
            with open(path, 'r') as meta_file:
                meta_data: 'dict[str, Any]' = json.load(meta_file)

        else:
            os.makedirs(self.data_dir, exist_ok=True)

            with open(path, 'w') as meta_file:
                json.dump({}, meta_file)

            meta_data = None

        # Load or create initial meta data
        if meta_data:
            self.meta = meta_data[sorted(meta_data.keys())[-1]]

        else:
            self.meta = {'ckpt_ctr': -1, 'ckpt_ver': 0, 'name': self.model_name}
            self.update()

        # Load checkpoint data and set RNG states
        if os.path.exists(path := self.meta['ckpt_path']):
            ckpt = torch.load(path)

            self.rng = np.random.default_rng()
            self.rng.__setstate__(ckpt['np_rng'])

            torch.set_rng_state(torch.tensor(ckpt['pt_rng'], dtype=torch.uint8))

            if self.device == 'cuda' and ckpt['pt_rng_cuda']:
                torch.cuda.set_rng_state(torch.tensor(ckpt['pt_rng_cuda'], dtype=torch.uint8))

        # Init. new RNG states
        else:
            if seed is None:
                seed = torch.initial_seed()

            self.rng = np.random.default_rng(seed)
            torch.random.manual_seed(seed)

        # Report on init or load event via log.txt file
        log_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if os.path.exists(log_path := os.path.join(self.data_dir, 'log.txt')):
            log_text = f'Resumed state from ckpt. {self.meta["ckpt_ver"]}.'

        else:
            log_text = f'Created state ckpt. {self.meta["ckpt_ver"]}.'

        with open(log_path, 'a') as log_file:
            log_file.write(f'{log_time} | {log_text}\n')

        print(log_text)

    def update(self, epoch_step: int = 0, update_step: int = 0, ckpt_increment: int = 0, score: float = None):
        self.meta['ckpt_ctr'] += 1
        self.meta['ckpt_ver'] += ckpt_increment
        self.meta['epoch_step'] = epoch_step
        self.meta['update_step'] = update_step
        self.meta['perf_score'] = score
        self.meta['model_path'] = os.path.join(self.data_dir, f'model_{self.meta["ckpt_ver"]:03d}.pt')
        self.meta['ckpt_path'] = os.path.join(self.data_dir, f'ckpt_{self.meta["ckpt_ver"]:03d}.pt')

    def checkpoint(self, epoch_step: int = 0, update_step: int = 0, ckpt_increment: int = 0, score: float = None):
        """Save current data."""

        # Update version and paths
        self.update(epoch_step, update_step, ckpt_increment, score)

        # Add to meta map
        with open(os.path.join(self.data_dir, 'meta.json'), 'r') as meta_file:
            meta_data = json.load(meta_file)

        meta_data[(log_time := datetime.now().strftime('%Y-%m-%d %H:%M:%S'))] = self.meta

        # Save meta backup
        with open(os.path.join(self.data_dir, 'meta_backup.json'), 'w') as meta_file:
            json.dump(meta_data, meta_file)

        # Save meta proper
        with open(os.path.join(self.data_dir, 'meta.json'), 'w') as meta_file:
            json.dump(meta_data, meta_file)

        # Save model params.
        model_state = None if self.model is None else self.model.state_dict()

        if model_state is not None:
            torch.save(model_state, self.meta['model_path'])

        # Save full checkpoint
        optim_state = None if self.optimiser is None else self.optimiser.state_dict()

        torch.save(
            {
                'model': model_state,
                'optim': optim_state,
                'np_rng': self.rng.__getstate__(),
                'pt_rng': torch.get_rng_state().tolist(),
                'pt_rng_cuda': torch.cuda.get_rng_state().tolist() if self.device == 'cuda' else [],
                **self.meta},
            self.meta['ckpt_path'])

        # Report on checkpoint event via log
        log_text = f'Saved ckpt. ver. {self.meta["ckpt_ver"]} on epoch {self.meta["epoch_step"]}.'

        with open(os.path.join(self.data_dir, 'log.txt'), 'a') as log_file:
            log_file.write(f'{log_time} | {log_text}\n')

        print(f'\n{log_text}')

    def load_model(self, model: Module, optimiser: Optimizer = None):
        """Restore model and optimiser params."""

        self.model = model.to(self.device)
        self.optimiser = optimiser

        if path_exists := os.path.exists(path := self.meta['ckpt_path']):
            ckpt = torch.load(path)

            if (state := ckpt['model']) is not None:
                model.load_state_dict(state)

            if optimiser is not None and (state := ckpt['optim']) is not None:
                optimiser.load_state_dict(state)

        print(f'{"Loaded" if path_exists else "Initialised"} model ver. {self.meta["ckpt_ver"]}.')
