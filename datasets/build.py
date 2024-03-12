# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates.

import os
import itertools
import logging
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.utils.data
import torch.utils.data as torchdata

import detectron2.utils.comm as comm
from detectron2.data.build import (
    build_batch_data_loader,
    load_proposals_into_dataset,
    trivial_batch_collator,
)
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    verify_results,
)
from fvcore.common.config import CfgNode

from .dataset_mappers import *
from .evaluation import (TextVQAEvaluator,
                         ScienceQAEvaluator,
                         POPEEvaluator,
                         MMEEvaluator,
                         MMBenchEvaluator,
                         QBenchEvaluator,
                         MMVetEvaluator,
                         TextVQAEvaluator,
                         ScienceQAEvaluator,
                         POPEEvaluator,
                         MMEEvaluator,
                         MMBenchEvaluator,
                         QBenchEvaluator,
                         MMVetEvaluator,
                         MathVistaEvaluator,
                         AI2DEvaluator,
                         HallusionBenchEvaluator,
                         SEEDEvaluator
)

from modeling.utils import configurable
from utils.distributed import get_world_size

def filter_images_with_only_crowd_annotations(dataset_dicts, dataset_names):
    """
    Filter out images with none annotations or only crowd annotations
    (i.e., images without non-crowd annotations).
    A common training-time preprocessing on COCO dataset.

    Args:
        dataset_dicts (list[dict]): annotations in Detectron2 Dataset format.

    Returns:
        list[dict]: the same format, but filtered.
    """
    num_before = len(dataset_dicts)

    def valid(anns):
        for ann in anns:
            if isinstance(ann, list):
                for instance in ann:
                    if instance.get("iscrowd", 0) == 0:
                        return True
            else:
                if ann.get("iscrowd", 0) == 0:
                    return True
        return False

    dataset_dicts = [x for x in dataset_dicts if valid(x["annotations"])]
    num_after = len(dataset_dicts)
    logger = logging.getLogger(__name__)
    logger.info(
        "Removed {} images with no usable annotations. {} images left.".format(
            num_before - num_after, num_after
        )
    )
    return dataset_dicts


def get_detection_dataset_dicts(
    dataset_names, filter_empty=True, proposal_files=None
):
    """
    Load and prepare dataset dicts for instance detection/segmentation and semantic segmentation.

    Args:
        dataset_names (str or list[str]): a dataset name or a list of dataset names
        filter_empty (bool): whether to filter out images without instance annotations
        proposal_files (list[str]): if given, a list of object proposal files
            that match each dataset in `dataset_names`.

    Returns:
        list[dict]: a list of dicts following the standard dataset dict format.
    """
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    assert len(dataset_names)
    
    dataset_dicts = [DatasetCatalog.get(dataset_name) for dataset_name in dataset_names]
    for dataset_name, dicts in zip(dataset_names, dataset_dicts):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)

    if proposal_files is not None:
        assert len(dataset_names) == len(proposal_files)
        # load precomputed proposals from proposal files
        dataset_dicts = [
            load_proposals_into_dataset(dataset_i_dicts, proposal_file)
            for dataset_i_dicts, proposal_file in zip(dataset_dicts, proposal_files)
        ]

    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))

    has_instances = "annotations" in dataset_dicts[0]
    if filter_empty and has_instances:
        dataset_dicts = filter_images_with_only_crowd_annotations(dataset_dicts, dataset_names)

    assert len(dataset_dicts), "No valid data found in {}.".format(",".join(dataset_names))
    return dataset_dicts


def _test_loader_from_config(cfg, dataset_name, mapper=None):
    """
    Uses the given `dataset_name` argument (instead of the names in cfg), because the
    standard practice is to evaluate each test set individually (not combining them).
    """
    if isinstance(dataset_name, str):
        dataset_name = [dataset_name]

    dataset = get_detection_dataset_dicts(
        dataset_name,
        filter_empty=False,
        proposal_files=None,
    )
    if mapper is None:
        mapper_cfg = CfgNode({'INPUT': cfg['INPUT'], 'MODEL': cfg['MODEL'], 'DATASETS': cfg['DATASETS']})
        mapper = DatasetMapper(mapper_cfg, False)
    assert cfg['TEST']['BATCH_SIZE_TOTAL'] % get_world_size() == 0, "Evaluation total batchsize is not divisible by gpu number"
    batch_size = cfg['TEST']['BATCH_SIZE_TOTAL'] // get_world_size()

    return {
        "dataset": dataset,
        "mapper": mapper,
        "num_workers": cfg['DATALOADER']['NUM_WORKERS'],
        "sampler": None,
        "batch_size": batch_size,
    }


@configurable(from_config=_test_loader_from_config)
def build_detection_test_loader(
    dataset: Union[List[Any], torchdata.Dataset],
    *,
    mapper: Callable[[Dict[str, Any]], Any],
    sampler: Optional[torchdata.Sampler] = None,
    batch_size: int = 1,
    num_workers: int = 0,
    collate_fn: Optional[Callable[[List[Any]], Any]] = None,
) -> torchdata.DataLoader:

    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    return torchdata.DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator if collate_fn is None else collate_fn,
    )


def _train_loader_from_config(cfg, dataset_name, mapper, *, dataset=None, sampler=None):
    cfg_datasets = cfg['DATASETS']
    cfg_dataloader = cfg['DATALOADER']
    
    if dataset is None:
        dataset = get_detection_dataset_dicts(
            dataset_name,
            filter_empty=cfg_dataloader['FILTER_EMPTY_ANNOTATIONS'],
            proposal_files=cfg_datasets['PROPOSAL_FILES_TRAIN'] if cfg_dataloader['LOAD_PROPOSALS'] else None,
        )

    if mapper is None:
        mapper = DatasetMapper(cfg, True)

    return {
        "dataset": dataset,
        "sampler": sampler,
        "mapper": mapper,
        "batch_size": cfg['TRAIN']['BATCH_SIZE_PER_GPU'],
        "aspect_ratio_grouping": cfg_dataloader['ASPECT_RATIO_GROUPING'],
        "num_workers": cfg_dataloader['NUM_WORKERS'],
    }


@configurable(from_config=_train_loader_from_config)
def build_detection_train_loader(
    dataset, *, mapper, sampler=None, batch_size, aspect_ratio_grouping=True, num_workers=0
):
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    return torchdata.DataLoader(
        dataset,
        shuffle=True,
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator
    )


def get_config_from_name(cfg, dataset_name):
    # adjust config according to dataset
    if 'sharegpt' in dataset_name:
        cfg.update(cfg['SHAREGPT'])
        return cfg
    elif 'textvqa' in dataset_name:
        cfg.update(cfg['TEXTVQA'])
        return cfg
    elif 'scienceqa' in dataset_name:
        cfg.update(cfg['SCIENCEQA'])
        return cfg
    elif 'pope' in dataset_name:
        cfg.update(cfg['POPE'])
        return cfg
    elif 'mme' in dataset_name:
        cfg.update(cfg['MME'])
        return cfg
    elif 'mmbench' in dataset_name:
        cfg.update(cfg['MMBENCH'])
        return cfg
    elif 'qbench' in dataset_name:
        cfg.update(cfg['QBENCH'])
        return cfg
    elif 'mm-vet' in dataset_name:
        cfg.update(cfg['MMVET'])
        return cfg
    elif 'mathvista' in dataset_name:
        cfg.update(cfg['MATHVISTA'])
        return cfg
    elif 'ai2d' in dataset_name:
        cfg.update(cfg['AI2D'])
        return cfg
    elif 'hallusionbench' in dataset_name:
        cfg.update(cfg['HALLUSIONBENCH'])
        return cfg
    elif 'seed' in dataset_name:
        cfg.update(cfg['SEED'])
        return cfg
    else:
        assert False, "dataset not support."


def build_eval_dataloader(cfg, ):
    dataset_name = cfg['DATASETS']['TEST']
    cfg = get_config_from_name(cfg, dataset_name)
    # adjust mapper according to dataset
    if dataset_name in ["sharegpt4v"]:
        mapper = ShareGPTDatasetMapper(cfg, False, dataset_name)
    elif dataset_name in ["textvqa_val"]:
        mapper = TextVQADatasetMapper(cfg, False, dataset_name)
    elif dataset_name in ["scienceqa_test"]:
        mapper = ScienceQADatasetMapper(cfg, False, dataset_name)
    elif dataset_name in ["pope_test"]:
        mapper = POPEDatasetMapper(cfg, False, dataset_name)
    elif dataset_name in ["mme"]:
        mapper = MMEDatasetMapper(cfg, False, dataset_name)
    elif dataset_name in ["mmbench", "mmbench_cn", "mmbench_test", "mmbench_test_cn"]:
        mapper = MMBenchDatasetMapper(cfg, False, dataset_name)
    elif dataset_name in ["qbench_test", "qbench_dev", "qbench_cn_test", "qbench_cn_dev"]:
        mapper = QBenchDatasetMapper(cfg, False, dataset_name)
    elif dataset_name in ["mm-vet"]:
        mapper = MMVetDatasetMapper(cfg, False, dataset_name)
    elif dataset_name in ["mathvista_testmini", "mathvista_test"]:
        mapper = MathVistaDatasetMapper(cfg, False, dataset_name)
    elif dataset_name in ["ai2d"]:
        mapper = AI2DDatasetMapper(cfg, False, dataset_name)
    elif dataset_name in ["hallusionbench"]:
        mapper = HallusionBenchDatasetMapper(cfg, False, dataset_name)
    elif dataset_name in ["seed"]:
        mapper = SEEDDatasetMapper(cfg, False, dataset_name)
    else:
        mapper = None
    return build_detection_test_loader(cfg, dataset_name, mapper=mapper)


def build_train_dataloader(cfg, ):
    dataset_name = cfg['DATASETS']['TRAIN']
    
    cfg = get_config_from_name(cfg, dataset_name)
    mapper_name = cfg['INPUT']['DATASET_MAPPER_NAME']
    # Semantic segmentation dataset mapper
    if mapper_name == "sharegpt":
        mapper = ShareGPTDatasetMapper(cfg, True, dataset_name)
        loaders = build_detection_train_loader(cfg, dataset_name=dataset_name, mapper=mapper)
    else:
        mapper = None
        loaders = build_detection_train_loader(cfg, dataset_name=dataset_name, mapper=mapper)

    return loaders

    
def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each
    builtin dataset. For your own dataset, you can simply create an
    evaluator manually in your script and do not have to worry about the
    hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg["SAVE_DIR"], "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

    # instance segmentation
    if evaluator_type == "textvqa":
        evaluator_list.append(TextVQAEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "scienceqa":
        evaluator_list.append(ScienceQAEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "pope":
        evaluator_list.append(POPEEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "mme":
        evaluator_list.append(MMEEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "mmbench":
        evaluator_list.append(MMBenchEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "qbench":
        evaluator_list.append(QBenchEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "mm-vet":
        evaluator_list.append(MMVetEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "mathvista":
        evaluator_list.append(MathVistaEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "ai2d":
        evaluator_list.append(AI2DEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "hallusionbench":
        evaluator_list.append(HallusionBenchEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "seed":
        evaluator_list.append(SEEDEvaluator(dataset_name, output_dir=output_folder))
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(
                dataset_name, evaluator_type
            )
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
        
    
    return DatasetEvaluators(evaluator_list)