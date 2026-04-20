#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import abc
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import draccus
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from lerobot.utils.constants import SCHEDULER_STATE
from lerobot.utils.import_utils import _diffusers_available, require_package
from lerobot.utils.io_utils import deserialize_json_into_object, write_json

if TYPE_CHECKING or _diffusers_available:
    from diffusers.optimization import get_scheduler
else:
    get_scheduler = None


@dataclass
class LRSchedulerConfig(draccus.ChoiceRegistry, abc.ABC):
    num_warmup_steps: int | None

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    @abc.abstractmethod
    def build(self, optimizer: Optimizer, num_training_steps: int) -> LRScheduler | None:
        raise NotImplementedError


@LRSchedulerConfig.register_subclass("diffuser")
@dataclass
class DiffuserSchedulerConfig(LRSchedulerConfig):
    name: str = "cosine"
    num_warmup_steps: int | None = None

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        require_package("diffusers", extra="diffusion")

        kwargs = {**asdict(self), "num_training_steps": num_training_steps, "optimizer": optimizer}
        return get_scheduler(**kwargs)


@LRSchedulerConfig.register_subclass("vqbet")
@dataclass
class VQBeTSchedulerConfig(LRSchedulerConfig):
    num_warmup_steps: int
    num_vqvae_training_steps: int
    num_cycles: float = 0.5

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        def lr_lambda(current_step):
            if current_step < self.num_vqvae_training_steps:
                return float(1)
            else:
                adjusted_step = current_step - self.num_vqvae_training_steps
                if adjusted_step < self.num_warmup_steps:
                    return float(adjusted_step) / float(max(1, self.num_warmup_steps))
                progress = float(adjusted_step - self.num_warmup_steps) / float(
                    max(1, num_training_steps - self.num_warmup_steps)
                )
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(self.num_cycles) * 2.0 * progress)))

        return LambdaLR(optimizer, lr_lambda, -1)


@LRSchedulerConfig.register_subclass("cosine_decay_with_warmup")
@dataclass
class CosineDecayWithWarmupSchedulerConfig(LRSchedulerConfig):
    """Used by Physical Intelligence to train Pi0.

    Automatically scales warmup and decay steps if num_training_steps < num_decay_steps.
    This ensures the learning rate schedule completes properly even with shorter training runs.
    """

    num_warmup_steps: int
    num_decay_steps: int
    peak_lr: float
    decay_lr: float

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        # Auto-scale scheduler parameters if training steps are shorter than configured decay steps
        actual_warmup_steps = self.num_warmup_steps
        actual_decay_steps = self.num_decay_steps

        if num_training_steps < self.num_decay_steps:
            # Calculate scaling factor to fit the schedule into the available training steps
            scale_factor = num_training_steps / self.num_decay_steps
            actual_warmup_steps = int(self.num_warmup_steps * scale_factor)
            actual_decay_steps = num_training_steps

            logging.info(
                f"Auto-scaling LR scheduler: "
                f"num_training_steps ({num_training_steps}) < num_decay_steps ({self.num_decay_steps}). "
                f"Scaling warmup: {self.num_warmup_steps} → {actual_warmup_steps}, "
                f"decay: {self.num_decay_steps} → {actual_decay_steps} "
                f"(scale factor: {scale_factor:.3f})"
            )

        def lr_lambda(current_step):
            def linear_warmup_schedule(current_step):
                if current_step <= 0:
                    return 1 / (actual_warmup_steps + 1)
                frac = 1 - current_step / actual_warmup_steps
                return (1 / (actual_warmup_steps + 1) - 1) * frac + 1

            def cosine_decay_schedule(current_step):
                step = min(current_step, actual_decay_steps)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * step / actual_decay_steps))
                alpha = self.decay_lr / self.peak_lr
                decayed = (1 - alpha) * cosine_decay + alpha
                return decayed

            if current_step < actual_warmup_steps:
                return linear_warmup_schedule(current_step)

            return cosine_decay_schedule(current_step)

        return LambdaLR(optimizer, lr_lambda, -1)


@LRSchedulerConfig.register_subclass("xvla_peft")
@dataclass
class XVLAPeftSchedulerConfig(LRSchedulerConfig):
    """Scheduler that reproduces X-VLA's group-wise PEFT LR logic.

    `num_warmup_steps` is the duration of the post-freeze warmup phase. For example,
    with `freeze_steps=1000` and `num_warmup_steps=1000`, steps 0-999 are frozen and
    steps 1000-1999 linearly warm up into joint training.
    """

    num_warmup_steps: int
    freeze_steps: int
    peak_lr: float
    learning_coef: float = 1.0
    min_lr_ratio: float = 0.1
    use_cosine_decay: bool = False

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        base_lrs: list[float] = []
        group_names: list[str] = []

        for group in optimizer.param_groups:
            name = group.get("name")
            if name == "vlm":
                base_lr = self.peak_lr * self.learning_coef
            elif name == "soft_prompts":
                base_lr = self.peak_lr * self.learning_coef
            else:
                base_lr = self.peak_lr

            group["lr"] = base_lr
            group_names.append(name)
            base_lrs.append(base_lr)

        def desired_lr(step: int, group_name: str) -> float:
            base_lr = self.peak_lr * self.learning_coef if group_name in {"vlm", "soft_prompts"} else self.peak_lr

            if step < self.freeze_steps:
                if group_name in {"vlm", "transformer_core"}:
                    return 0.0
                return base_lr

            if not self.use_cosine_decay:
                return base_lr

            progress = step - self.freeze_steps
            if progress < self.num_warmup_steps:
                return base_lr * (progress / max(1, self.num_warmup_steps))

            remain = max(1, num_training_steps - (self.freeze_steps + self.num_warmup_steps))
            ratio = 0.5 * (1 + math.cos(math.pi * min(1.0, (progress - self.num_warmup_steps) / remain)))
            return base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * ratio)

        lr_lambdas = []
        for base_lr, group_name in zip(base_lrs, group_names, strict=True):
            if base_lr == 0.0:
                lr_lambdas.append(lambda current_step: 1.0)
                continue

            # The scheduler is stepped after the optimizer update in LeRobot, so
            # the next batch should observe step=current_step+1.
            lr_lambdas.append(
                lambda current_step, group_name=group_name, base_lr=base_lr: desired_lr(
                    current_step + 1, group_name
                )
                / base_lr
            )

        scheduler = LambdaLR(optimizer, lr_lambdas, -1)

        initial_lrs = [desired_lr(0, group_name) for group_name in group_names]
        for group, initial_lr in zip(optimizer.param_groups, initial_lrs, strict=True):
            group["lr"] = initial_lr
        scheduler._last_lr = initial_lrs

        return scheduler


def save_scheduler_state(scheduler: LRScheduler, save_dir: Path) -> None:
    state_dict = scheduler.state_dict()
    write_json(state_dict, save_dir / SCHEDULER_STATE)


def load_scheduler_state(scheduler: LRScheduler, save_dir: Path) -> LRScheduler:
    state_dict = deserialize_json_into_object(save_dir / SCHEDULER_STATE, scheduler.state_dict())
    scheduler.load_state_dict(state_dict)
    return scheduler
