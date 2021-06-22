import contextlib
import copy
import io
import itertools
import json
import os

import numpy as np
from pycocotools.cocoeval import COCOeval

from cvpods.evaluation import COCOEvaluator
from cvpods.evaluation.registry import EVALUATOR
from cvpods.utils import PathManager

from .metadata import _get_coco_fewshot_instances_meta


class COCOFewShotEvaluator(COCOEvaluator):
    def __init__(self, dataset_name, metadata, cfg,
                 distributed, output_dir=None, dump=False):
        super().__init__(dataset_name, metadata, cfg,
                         distributed, output_dir, dump)
        assert any(n in dataset_name for n in ["all", "base", "novel"]), \
            "Few-shot datasets must contain few-shot keywords"
        self._dataset_name = dataset_name
        origin_metadata = _get_coco_fewshot_instances_meta()
        self._base_classes = sorted(list(
            origin_metadata["base_dataset_id_to_contiguous_id"].keys()))
        self._novel_classes = sorted(list(
            origin_metadata["novel_dataset_id_to_contiguous_id"].keys()))

    def _eval_predictions(self, tasks):
        self._logger.info("Preparing results for COCO format (few-shot)...")
        self._coco_results = list(itertools.chain(
            *[x["instances"] for x in self._predictions]
        ))

        all_class_names = self._metadata.thing_classes
        base_class_names = self._metadata.base_classes
        novel_class_names = self._metadata.novel_classes

        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k for k, v in
                self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            for result in self._coco_results:
                category_id = result["category_id"]
                assert (
                    category_id in reverse_id_mapping
                ), "A prediction has category_id={}, \
                    which is not available in the dataset".format(
                    category_id
                )
                result["category_id"] = reverse_id_mapping[category_id]

        if self._output_dir:
            file_path = os.path.join(
                    self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating predictions ...")
        for task in sorted(tasks):
            self._results[task] = {}
            for split, classes, names in [
                ("all", None, all_class_names),
                ("base", self._base_classes, base_class_names),
                ("novel", self._novel_classes, novel_class_names)
            ]:
                if "all" not in self._dataset_name and \
                        split not in self._dataset_name:
                    continue
                coco_eval, summary = (
                    _evaluate_predictions_on_coco(
                        self._coco_api, self._coco_results,
                        task, catIds=classes,
                    )
                    if len(self._coco_results) > 0
                    else None
                )
                res_ = self._derive_coco_results(
                    coco_eval, task, summary, class_names=names,
                )
                res = {}
                for metric in res_.keys():
                    if len(metric) <= 4:
                        if split == "all":
                            res[metric] = res_[metric]
                        elif split == "base":
                            res["b" + metric] = res_[metric]
                        elif split == "novel":
                            res["n" + metric] = res_[metric]
                self._results[task].update(res)
            if "AP" not in self._results[task]:
                if "nAP" in self._results["bbox"]:
                    self._results[task]["AP"] = self._results[task]["nAP"]
                else:
                    self._results[task]["AP"] = self._results[task]["bAP"]


def _evaluate_predictions_on_coco(
        coco_gt, coco_results, iou_type, kpt_oks_sigmas=None, catIds=None):
    """
    This evaluation function supports the selection of catIds params.
    """
    assert len(coco_results) > 0
    if iou_type == "segm":
        coco_results = copy.deepcopy(coco_results)
        for c in coco_results:
            c.pop("bbox", None)

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    if kpt_oks_sigmas:
        coco_eval.params.kpt_oks_sigmas = np.array(kpt_oks_sigmas)
    # Cat ids are used for few-shot evaluation
    if catIds is not None:
        coco_eval.params.catIds = catIds

    coco_eval.evaluate()
    coco_eval.accumulate()

    redirect_string = io.StringIO()
    with contextlib.redirect_stdout(redirect_string):
        coco_eval.summarize()
    redirect_string.getvalue()
    return coco_eval, redirect_string


EVALUATOR._obj_map["COCOEvaluator"] = COCOFewShotEvaluator
