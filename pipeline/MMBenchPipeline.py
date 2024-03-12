import torch
import logging
from tqdm import tqdm
from typing import Tuple, Dict

from trainer.default_trainer import DefaultTrainer
from modeling import build_model
from modeling.BaseModel import BaseModel
from datasets import build_evaluator, build_eval_dataloader, build_train_dataloader
from trainer.utils.misc import move_batch_to_device

from .utils.misc import hook_opt
import torch.distributed as dist
from moai.utils.utils import *

logger = logging.getLogger(__name__)

class MMBenchPipeline:
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
            self.evaluator_total = build_evaluator(self._opt, self._opt['DATASETS']['TEST'], self._opt['SAVE_DIR'])
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
    def eval_freeze(model):
        model.eval()
        for param in model.parameters(): param.requires_grad = False

    @staticmethod
    def print_dtype(model):
        for param in model.parameters(): print(param.dtype)

    def evaluate_model(
        self,
        trainer: DefaultTrainer,
    ) -> Tuple[Dict, Dict[str, float], bool]:

        self._opt = hook_opt(self._opt)
        scores = {}
        
        torch.cuda.empty_cache()
        eval_batch_gen = self.get_dataloaders(trainer, self._opt['DATASETS']['TEST'], is_evaluation=True)
        self.evaluator_total.reset()
        
        # accelerate wrapping
        model, eval_batch_gen = trainer.accel.prepare(trainer.model.model, eval_batch_gen)

        # DDP module unwrapping
        try:
            model = model.module
        except:
            pass
        
        # eval mode
        model.eval()

        # Language
        if "cn" in self._opt['DATASETS']['TEST']:
            language_prompt = "请直接回答选项字母。"
        else:
            language_prompt = "Answer with the option's letter from the given choices directly."
        
        with torch.no_grad():
            prog_bar = tqdm(enumerate(eval_batch_gen), total=len(eval_batch_gen), leave=True, disable=not trainer.accel.is_local_main_process)
            for idx, batch in prog_bar:

                batch = move_batch_to_device(batch, trainer.accel.device)

                # MoAI: llm preparation
                moai_inputs = model.moai_model.eval_process(images=torch.stack([x['image'] for x in batch]), 
                                                                    prompts=[f"{x['question']}\n{language_prompt}" for x in batch], 
                                                                    processor=model.moai_processor,
                                                                    seg_model=model.seg_model,
                                                                    seg_processor=model.seg_processor,
                                                                    od_model=model.od_model,
                                                                    od_processor=model.od_processor,
                                                                    sgg_model=model.sgg_model,
                                                                    ocr_model=model.ocr_model,
                                                                    device=trainer.accel.device,
                                                                    mode='moai_eval.yaml')
                
                # Batch Generate
                with torch.inference_mode():
                    generate_ids = model.moai_model.generate(**moai_inputs, do_sample=False, num_beams=3, max_new_tokens=128, use_cache=True)
                
                # Batch Decoding
                decoded_text = []
                for gen_id in generate_ids:
                    decoded_text.append(model.moai_processor.batch_decode(gen_id[gen_id!=-100].unsqueeze(0), skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])


                # Batch VQA evaluate process
                self.evaluator_total.process(batch, {'question_id': [x["question_id"] for x in batch], 
                                                        'question': [x["question"] for x in batch], 
                                                    'text': [x.split('assistant\n')[-1].strip().split('[UN')[0] for x in decoded_text]})
                # garbage collection
                torch.cuda.empty_cache()

        # Total Result Write on CSV
        results = self.evaluator_total.evaluate()
        if trainer.accel.is_main_process: print(results)
        return scores