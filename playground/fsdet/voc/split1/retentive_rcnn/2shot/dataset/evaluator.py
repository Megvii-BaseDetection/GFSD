import os
import tempfile
from collections import OrderedDict, defaultdict

import numpy as np

from cvpods.evaluation import PascalVOCDetectionEvaluator
from cvpods.evaluation.pascal_voc_evaluation import (_dump_to_markdown,
                                                      create_small_table,
                                                      voc_eval)
from cvpods.evaluation.registry import EVALUATOR
from cvpods.utils import comm

from .dataset import PASCAL_VOC_BASE_CATEGORIES, PASCAL_VOC_NOVEL_CATEGORIES


class PascalVOCFewShotDetectionEvaluator(
    PascalVOCDetectionEvaluator
):
    def __init__(self, dataset_name, meta, dump=False):
        super().__init__(dataset_name, meta, dump)
        few_shot_keywords = ["all", "novel", "base"]
        datasplit = [n for n in few_shot_keywords if n in dataset_name]
        assert len(datasplit) == 1
        sid = int(dataset_name[-1])
        datasplit = datasplit[0]
        self._base_classes = None if datasplit == "novel" else \
            PASCAL_VOC_BASE_CATEGORIES[sid]
        self._novel_classes = None if datasplit == "base" else \
            PASCAL_VOC_NOVEL_CATEGORIES[sid]

    def evaluate(self):
        all_predictions = comm.gather(self._predictions, dst=0)
        if not comm.is_main_process():
            return
        predictions = defaultdict(list)
        for predictions_per_rank in all_predictions:
            for clsid, lines in predictions_per_rank.items():
                predictions[clsid].extend(lines)
        del all_predictions

        self._logger.info(
            "Evaluating {} using {} metric. "
            "Note that results do not "
            "use the official Matlab API.".format(
                self._dataset_name, 2007 if self._is_2007 else 2012
            )
        )

        with tempfile.TemporaryDirectory(
                prefix="pascal_voc_eval_") as dirname:
            res_file_template = os.path.join(dirname, "{}.txt")

            aps = defaultdict(list)  # iou -> ap per class
            aps_base = defaultdict(list)
            aps_novel = defaultdict(list)
            exist_base, exist_novel = False, False
            for cls_id, cls_name in enumerate(self._class_names):
                lines = predictions.get(cls_id, [""])

                with open(res_file_template.format(cls_name),
                          "w") as f:
                    f.write("\n".join(lines))

                for thresh in range(50, 100, 5):
                    rec, prec, ap = voc_eval(
                        res_file_template,
                        self._anno_file_template,
                        self._image_set_path,
                        cls_name,
                        ovthresh=thresh / 100.0,
                        use_07_metric=self._is_2007,
                    )
                    aps[thresh].append(ap * 100)

                    if self._base_classes is not None and \
                            cls_name in self._base_classes:
                        aps_base[thresh].append(ap * 100)
                        exist_base = True

                    if self._novel_classes is not None and \
                            cls_name in self._novel_classes:
                        aps_novel[thresh].append(ap * 100)
                        exist_novel = True

        ret = OrderedDict()
        mAP = {iou: np.mean(x) for iou, x in aps.items()}
        ret["bbox"] = {"AP": np.mean(list(mAP.values())),
                       "AP50": mAP[50], "AP75": mAP[75]}

        # adding evaluation of the base and novel classes
        if exist_base:
            mAP_base = {iou: np.mean(x)
                        for iou, x in aps_base.items()}
            ret["bbox"].update(
                {"bAP": np.mean(list(mAP_base.values())),
                 "bAP50": mAP_base[50],
                 "bAP75": mAP_base[75]}
            )

        if exist_novel:
            mAP_novel = {iou: np.mean(x)
                         for iou, x in aps_novel.items()}
            ret["bbox"].update({
                "nAP": np.mean(list(mAP_novel.values())),
                "nAP50": mAP_novel[50],
                "nAP75": mAP_novel[75]
            })

        # write per class AP to logger
        per_class_res = {self._class_names[idx]: ap
                         for idx, ap in enumerate(aps[50])}
        per_class_table = create_small_table(per_class_res)
        small_table = create_small_table(ret["bbox"])
        self._logger.info("Evaluate per-class mAP50:\n"
                          + per_class_table)
        self._logger.info("Evaluate overall bbox:\n"
                          + small_table)

        if self._dump:
            dump_info_one_task = {
                "task": "bbox",
                "tables": [per_class_table, small_table],
            }
            _dump_to_markdown([dump_info_one_task])
        return ret


EVALUATOR._obj_map["PascalVOCDetectionEvaluator"] = \
    PascalVOCFewShotDetectionEvaluator
