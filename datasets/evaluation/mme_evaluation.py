import os
import json
import logging
import itertools

import detectron2.utils.comm as comm

from detectron2.evaluation.evaluator import DatasetEvaluator
from collections import defaultdict
from moai.eval.m4c_evaluator import mme_calculate_metrics

_root = os.getenv("DATASET2", "datasets")

class MMEEvaluator(DatasetEvaluator):
    """
    Evaluate MME Accuracy
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

    def get_gt(self, data_path):
        GT = {}
        for category in os.listdir(data_path):
            category_dir = os.path.join(data_path, category)
            if not os.path.isdir(category_dir):
                continue
            if os.path.exists(os.path.join(category_dir, 'images')):
                image_path = os.path.join(category_dir, 'images')
                qa_path = os.path.join(category_dir, 'questions_answers_YN')
            else:
                image_path = qa_path = category_dir
            assert os.path.isdir(image_path), image_path
            assert os.path.isdir(qa_path), qa_path
            for file in os.listdir(qa_path):
                if not file.endswith('.txt'):
                    continue
                for line in open(os.path.join(qa_path, file)):
                    question, answer = line.strip().split('\t')
                    GT[(category, file, question)] = answer
        return GT

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

        pred_answers = [{"question_id": question_id, "answer": answer, "question": question, 'category': batch_input['category']} for question_id, answer, question, batch_input in zip(question_ids, gen_answers,questions, inputs)]
        pred_pth = os.path.join(self._output_dir, '{}_results.json'.format(self._dataset_name))
        json.dump(pred_answers, open(pred_pth, "w"))


        predictions = {pred['question_id']: pred for pred in pred_answers}

        GT = self.get_gt(
            data_path=os.path.join(_root, 'MME_Benchmark_release_version')
        )

        result_dir = os.path.join(self._output_dir, 'mme')
        os.makedirs(result_dir, exist_ok=True)

        answers = pred_answers

        results = defaultdict(list)
        for answer in answers:
            category = answer['question_id'].split('/')[0]
            file = answer['question_id'].split('/')[-1].split('.')[0] + '.txt'
            question = answer['question']
            results[category].append((file, answer['question'], answer['answer']))

        for category, cate_tups in results.items():
            with open(os.path.join(result_dir, f'{category}.txt'), 'w') as fp:
                for file, prompt, answer in cate_tups:
                    if 'Answer the question using a single word or phrase.' in prompt:
                        prompt = prompt.replace('Answer the question using a single word or phrase.', '').strip()
                    if 'Please answer yes or no.' not in prompt:
                        prompt = prompt + ' Please answer yes or no.'
                        if (category, file, prompt) not in GT:
                            prompt = prompt.replace(' Please answer yes or no.', '  Please answer yes or no.')
                    gt_ans = GT[category, file, prompt]
                    tup = file, prompt, gt_ans, answer
                    fp.write('\t'.join(tup) + '\n')

        cal = mme_calculate_metrics()
        scores = cal.process_result(result_dir)
        
        return scores