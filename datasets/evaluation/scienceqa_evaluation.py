import os
import re
import json
import logging
import itertools

import detectron2.utils.comm as comm

from detectron2.evaluation.evaluator import DatasetEvaluator

_root = os.getenv("DATASET2", "datasets")

class ScienceQAEvaluator(DatasetEvaluator):
    """
    Evaluate ScienceQA Accuracy
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
            self._gen_answers.append(x.strip())
            self._question_ids.append(y)
            self._questions.append(z)
    
    def convert_caps(self, results):
        fakecaps = []
        for result in results:
            image_id = result['question_id']
            caption = result['text']
            fakecaps.append({"image_id": int(image_id), "caption": caption})
        return fakecaps


    def get_pred_idx(self, prediction, choices, options):
        """
        Get the index (e.g. 2) from the prediction (e.g. 'C')
        """
        if prediction in options[:len(choices)]:
            return options.index(prediction)
        else:
            return -1        

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

        results = {'correct': [], 'incorrect': []}
        sqa_results = {}
        sqa_results['acc'] = None
        sqa_results['correct'] = None
        sqa_results['count'] = None
        sqa_results['results'] = {}
        sqa_results['outputs'] = {}

        pred_answers = [{"question_id": question_id, "answer": answer, "question": question} for question_id, answer, question in zip(question_ids, gen_answers,questions)]
        pred_pth = os.path.join(self._output_dir, '{}_results.json'.format(self._dataset_name))
        json.dump(pred_answers, open(pred_pth, "w"))

        annotation_file = os.path.join(_root, 'ScienceQA/problems.json')
        annotations = json.load(open(annotation_file))

        predictions = {pred['question_id']: pred for pred in pred_answers}
        options = ["A", "B", "C", "D", "E"]

        for prob_id, prob in annotations.items():
            if prob_id not in predictions:
                pred = {'text': 'FAILED', 'question': 'Unknown'}
                pred_text = 'FAILED'
                continue # ignore questions without images
            else:
                pred = predictions[prob_id]
                pred_text = pred['answer']

            if pred_text in options:
                answer = pred_text
            elif len(pred_text) >= 3 and pred_text[0] in options and pred_text[1:3] == ". ":
                answer = pred_text[0]
            else:
                pattern = re.compile(r'The answer is ([A-Z]).')
                res = pattern.findall(pred_text)
                if len(res) == 1:
                    answer = res[0]  # 'A', 'B', ...
                else:
                    answer = "FAILED"

            pred_idx = self.get_pred_idx(answer, prob['choices'], options)

            analysis = {
                'question_id': prob_id,
                'parsed_ans': answer,
                'ground_truth':options[prob['answer']],
                'question': pred['question'],
                'pred': pred_text,
                'is_multimodal': '<image>' in pred['question'],
            }

            sqa_results['results'][prob_id] = self.get_pred_idx(answer, prob['choices'], options)
            sqa_results['outputs'][prob_id] = pred_text

            if pred_idx == prob['answer']:
                results['correct'].append(analysis)
            else:
                results['incorrect'].append(analysis)

        correct = len(results['correct'])
        total = len(results['correct']) + len(results['incorrect'])

        ###### IMG ######
        multimodal_correct = len([x for x in results['correct'] if x['is_multimodal']])
        multimodal_incorrect = len([x for x in results['incorrect'] if x['is_multimodal']])
        multimodal_total = multimodal_correct + multimodal_incorrect
        # ###### IMG ######

        print(f'Total: {total}, Correct: {correct}, Accuracy: {correct / total * 100:.2f}%, IMG-Accuracy: {multimodal_correct / multimodal_total * 100:.2f}%')

        sqa_results['accuracy'] = correct / total * 100
        sqa_results['image_accuracy'] = multimodal_correct / multimodal_total * 100
        sqa_results['correct'] = correct
        sqa_results['count'] = total

        return sqa_results