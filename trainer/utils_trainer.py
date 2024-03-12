# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

from datetime import datetime
import os
import logging
import torch

from .distributed_trainer import DistributedTrainer
from .utils.misc import *
from utils.distributed import get_world_size

logger = logging.getLogger(__name__)


class UtilsTrainer(DistributedTrainer):

    def __init__(self, opt):
        super().__init__(opt)

    def get_batch_size(self, batch):
        if hasattr(self.model, 'get_batch_size'):
            if callable(self.model.get_batch_size):
                return self.model.get_batch_size(batch)
        return {}

    # Deepspeed & DDP compatible
    def _initialize_accelerator(self):
        if self.accel.state.deepspeed_plugin is not None:
            self.accel.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = self.opt['COCO']['TRAIN']['BATCH_SIZE_PER_GPU']
        self.model, self.optimizer, self.lr_scheduler, self.train_dataloaders = \
            self.accel.prepare(self.model, self.optimizer, self.lr_scheduler, self.train_dataloaders)
        if self.accel.state.deepspeed_plugin is None: self.model = self.model.module

    def save_checkpoint(self, epoch):
        
        save_dir = self.save_folder

        if torch.distributed.get_world_size() > 1:
            torch.distributed.barrier()

        self.model.save_pretrained(save_dir, epoch, self.accel)
        if self.accel.is_main_process:
            print(f'Saved!: {save_dir}')
        
        if torch.distributed.get_world_size() > 1:
            torch.distributed.barrier()