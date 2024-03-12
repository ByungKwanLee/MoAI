# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

import torch
from .default_trainer import DefaultTrainer

class MoAI_Trainer(DefaultTrainer):
    def create_optimizer_and_scheduler(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(self.opt['OPTIMIZER']['LR']), weight_decay=self.opt['OPTIMIZER']['WEIGHT_DECAY'])
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=len(self.train_dataloaders)*self.opt['OPTIMIZER']['EPOCH'], eta_min=float(self.opt['OPTIMIZER']['LAST_LR']))
