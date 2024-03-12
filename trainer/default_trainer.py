# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

from datetime import datetime
import os
import sys
import importlib
import wandb
import logging
import torch

from .distributed_trainer import DistributedTrainer
from .utils_trainer import UtilsTrainer
from .utils.misc import *
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DefaultTrainer(UtilsTrainer, DistributedTrainer):

    def __init__(self, opt):
        """
        Set up the task the model is being trained for.
        """
        super().__init__(opt)
        base_name = 'base_dir'
        base_path =  os.path.join(self.opt['base_path'], '__init__.py')
        spec = importlib.util.spec_from_file_location(base_name, base_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[base_name] = module
        spec.loader.exec_module(module)
        # logger.info(f"Imported {base_name} at base_path {self.opt['base_path']}")

        pipeline_module = importlib.import_module(f"base_dir.pipeline.{self.opt['PIPELINE']}")
        pipeline_class = getattr(pipeline_module, self.opt['PIPELINE'])
        # logger.info(f"Pipeline for training: {self.opt['PIPELINE']}")
        self.pipeline = pipeline_class(self.opt)

    def eval(self):
        self.mode = "eval"

        self.model = self.pipeline.initialize_model()

        # move model to the device
        self.model.to(self.accel.device)

        results = self._eval_on_set()
        return results
    
    
    def _eval_on_set(self):      
        results = self.pipeline.evaluate_model(self)
        return results