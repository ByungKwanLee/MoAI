import os
import json
import logging
import itertools

import detectron2.utils.comm as comm
import pandas as pd

from detectron2.evaluation.evaluator import DatasetEvaluator

_root = os.getenv("DATASET2", "datasets")

class MMBenchEvaluator(DatasetEvaluator):
    """
    Evaluate MMBench Accuracy
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
        self._inputs = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the batch inputs to SysLearner model.
            outputs: the outputs of a SysLearner model. It is a list of dicts with key
                "text" that contains generated answers and "question_id" that contains question ids.
        """
        for x, y, z, w in zip(inputs, outputs['text'], outputs['question_id'], outputs['question']):
            self._inputs.append(x)
            self._gen_answers.append(y)
            self._question_ids.append(z)
            self._questions.append(w.strip())
    
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
            inputs = gather(self._inputs)
            if not comm.is_main_process():
                return {}
        else:
            gen_answers = self._gen_answers
            question_ids = self._question_ids
            questions = self._questions
            inputs = self._inputs

        pred_answers = [{"question_id": question_id, "answer": answer, "question": question, "round_id": batch_input['round_id']} for question_id, answer, question, batch_input in zip(question_ids, gen_answers,questions, inputs)]
        pred_pth = os.path.join(self._output_dir, '{}_results.json'.format(self._dataset_name))
        json.dump(pred_answers, open(pred_pth, "w"))

        if "test" in self._dataset_name:
            if "cn" in self._dataset_name:
                df = pd.read_table(os.path.join(_root, 'MMBench/MMBench_TEST_CN_legacy.tsv'))
            else:
                df = pd.read_table(os.path.join(_root, 'MMBench/MMBench_TEST_EN_legacy.tsv'))
        else:
            if "cn" in self._dataset_name:
                df = pd.read_table(os.path.join(_root, 'MMBench/mmbench_dev_cn_20231003.tsv'))
            else:
                df = pd.read_table(os.path.join(_root, 'MMBench/mmbench_dev_20230712.tsv'))

        cur_df = df.copy()
        cur_df = cur_df.drop(columns=['hint', 'category', 'source', 'image', 'comment', 'l2-category'])
        cur_df.insert(6, 'prediction', None)
        for pred in pred_answers:
            cur_df.loc[df['index'] == pred['question_id'], 'prediction'] = pred['answer']

        cur_df.to_excel(os.path.join(self._output_dir, f"{self._dataset_name}.xlsx"), index=False, engine='openpyxl')
        return