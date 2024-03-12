# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

import json
import logging

import torch
from tqdm import tqdm
from typing import Tuple, Dict

from trainer.default_trainer import DefaultTrainer

from modeling import build_model
from modeling.BaseModel import BaseModel
from datasets import build_eval_dataloader, build_train_dataloader
from trainer.utils.misc import move_batch_to_device
import torch.distributed as dist

logger = logging.getLogger(__name__)


class MoAIPipeline:
    def __init__(self, opt):
        self._opt = opt

    def initialize_model(self):
        model = build_model(self._opt)
        model.train()
        model = BaseModel(self._opt, model)
        return model

    def get_dataloaders(
        self, trainer: DefaultTrainer,
        dataset_label: str,
        is_evaluation: bool
    ):
        distributed = self._opt['world_size'] > 1
        if is_evaluation:
            if not hasattr(self, 'valid_loader'):
                dataloaders = build_eval_dataloader(self._opt)
                self.valid_loader = dataloaders
            else:
                dataloaders = self.valid_loader
            dataloader = dataloaders
        else:
            if not hasattr(self, 'train_loader'):
                dataloader = build_train_dataloader(self._opt)
                self.train_loader = dataloader
                # logger.info(f'num of train samples: {len(dataloader)}')
            else:
                dataloader = self.train_loader
                
            # temp solution for lr scheduler
            steps_total = len(self.train_loader)
            steps_acc = self._opt['OPTIMIZER']['GRAD_CUM']
            steps_update = steps_total // steps_acc
            self._opt["LR_SCHEDULER_PARAMS"]["steps_update_per_epoch"] = steps_update
        return dataloader

    @staticmethod
    def all_gather(data, world_size):
        output = [None for _ in range(world_size)]
        dist.all_gather_object(output, data, group=None)
        return output

    @staticmethod
    def forward_func(trainer, batch):
        loss = trainer.model(batch, trainer.accel)
        return loss