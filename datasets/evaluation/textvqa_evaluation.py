import os
import re
import json
import logging
import itertools

import detectron2.utils.comm as comm
from detectron2.evaluation.evaluator import DatasetEvaluator

from moai.eval.m4c_evaluator import TextVQAAccuracyEvaluator

_root = os.getenv("DATASET2", "datasets")

class TextVQAEvaluator(DatasetEvaluator):
    """
    Evaluate TextVQA Accuracy
    """

    def __init__(
        self,
        dataset_name=None,
        distributed=True,
        output_dir=None,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:
                    "json_file": the path to the COCO format annotation
                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            distributed (True): if True, will collect results from all ranks and run evaluation
                in the main process.
                Otherwise, will only evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:
                1. "instances_predictions.pth" a file that can be loaded with `torch.load` and
                   contains all the results in the format they are produced by the model.
                2. "coco_instances_results.json" a json file in COCO's result format.
        """
        self._logger = logging.getLogger(__name__)
        self._distributed = distributed
        self._output_dir = output_dir
        self._dataset_name = dataset_name
        self._output_dir = output_dir

    def reset(self):
        self._gen_answers = []
        self._question_ids = []
        self._questions = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the batch inputs to SysLearner model.
            outputs: the outputs of a SysLearner model. It is a list of dicts with key
                "text" that contains generated answers and "question_id" that contains question ids.
        """
        for x,y,z in zip(outputs['text'], outputs['question_id'], outputs['question']):
            self._gen_answers.append(x.strip().lower())
            self._question_ids.append(y)
            self._questions.append(z)
    
    def prompt_processor(self, prompt):
        if prompt.startswith('OCR tokens: '):
            pattern = r"Question: (.*?) Short answer:"
            match = re.search(pattern, prompt, re.DOTALL)
            question = match.group(1)
        elif 'Reference OCR token: ' in prompt and len(prompt.split('\n')) == 3:
            if prompt.startswith('Reference OCR token:'):
                question = prompt.split('\n')[1]
            else:
                question = prompt.split('\n')[0]
        elif len(prompt.split('\n')) == 2:
            question = prompt.split('\n')[0]
        else:
            assert False

        return question.lower()
    

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            def gather(x, move=False):
                x = comm.gather(x)
                x = list(itertools.chain(*x))
                if move:
                    x = [xx.to(self._gen_answers[0].device) for xx in x]
                return x
            gen_answers = gather(self._gen_answers)
            question_ids = gather(self._question_ids)
            questions = gather(self._questions)
            if not comm.is_main_process():
                return {}
        else:
            gen_answers = self._gen_answers
            question_ids = self._question_ids
            questions = self._questions

        pred_answers = [{"question_id": question_id, "answer": answer, "question": question} for question_id, answer, question in zip(question_ids, gen_answers,questions)]
        pred_pth = os.path.join(self._output_dir, '{}_results.json'.format(self._dataset_name))
        json.dump(pred_answers, open(pred_pth, "w"))

        result_file = pred_pth
        annotation_file = os.path.join(_root, 'TextVQA/TextVQA_0.5.1_val.json')
        experiment_name = os.path.splitext(os.path.basename(result_file))[0]
        print(experiment_name)
        annotations = json.load(open(annotation_file))['data']
        annotations = {(annotation['image_id'], annotation['question'].lower()): annotation for annotation in annotations}
        results = json.load(open(result_file))

        pred_list = []
        for result in results:
            annotation = annotations[(result['question_id'], self.prompt_processor(result['question']))]
            pred_list.append({
                "pred_answer": result['answer'],
                "gt_answers": annotation['answers'],
            })
            

        evaluator = TextVQAAccuracyEvaluator()
        print('Samples: {}\nAccuracy: {:.2f}%\n'.format(len(pred_list), 100. * evaluator.eval_pred_list(pred_list)))

        return {"accuracy": 100. * evaluator.eval_pred_list(pred_list)}