import contextlib
import io
import logging
import os

import numpy as np

from cvpods.data.datasets.builtin_meta import _get_builtin_metadata
from cvpods.data.datasets.coco import COCODataset
from cvpods.data.registry import DATASETS
from cvpods.structures import BoxMode
from cvpods.utils import PathManager, Timer

from .metadata import _get_coco_fewshot_instances_meta as get_fewshot_meta
from .paths_route import _PREDEFINED_SPLITS_COCO_FEWSHOT

logger = logging.getLogger(__name__)


@DATASETS.register()
class COCOFewShotDataset(COCODataset):
    """
    Dealing with few shot data splits and class-splited coco dataset.
    """
    def __init__(self, cfg, dataset_name, transforms=[], is_train=True):
        self.merge_few_shot_splits = getattr(
            cfg.DATASETS, "MERGE_FEW_SHOT_SPLITS", False)
        super().__init__(cfg, dataset_name, transforms, is_train)

    def _get_metadata(self):
        if "_all" in self.name:
            metadata = _get_builtin_metadata("coco")
        else:
            metadata = get_fewshot_meta()
            split = "base" if "_base" in self.name else "novel"
            metadata["thing_dataset_id_to_contiguous_id"] = \
                metadata["{}_dataset_id_to_contiguous_id".format(split)]
            metadata["thing_classes"] = metadata["{}_classes".format(split)]

        metadata["base_classes"] = get_fewshot_meta()["base_classes"]
        metadata["novel_classes"] = get_fewshot_meta()["novel_classes"]

        image_root, json_file = \
            _PREDEFINED_SPLITS_COCO_FEWSHOT["coco"][self.name]
        metadata["image_root"] = os.path.join(self.data_root, image_root) \
            if "://" not in image_root else image_root
        metadata["json_file"] = os.path.join(self.data_root, json_file) \
            if "://" not in image_root else os.path.join(image_root, json_file)
        metadata["evaluator_type"] = \
            _PREDEFINED_SPLITS_COCO_FEWSHOT["evaluator_type"]["coco"]
        return metadata

    def _merge_dataset_dicts(self, dataset_dicts=None):
        """
        Merge annotation-wise dataset dicts to image-wise dataset dicts.
        Note that TFA uses annotation-wise dataset for low-shot data.
        Annotation-wise dataset might be beneficial for 2-stage
        detectors like RCNN (with ~0.4 10shot mAP improvement) but harmful for
        1-stage detectors like FCOS.
        """
        dataset_dicts = dataset_dicts or self.dataset_dicts
        new_dataset_dicts = {}
        for d in dataset_dicts:
            file_name = d["file_name"]
            if file_name in new_dataset_dicts:
                new_dataset_dicts[file_name]["annotations"].extend(
                    d["annotations"])
            else:
                new_dataset_dicts[file_name] = d
        return list(new_dataset_dicts.values())

    def _load_annotations(self,
                          json_file,
                          image_root,
                          dataset_name=None,
                          extra_annotation_keys=None):
        from pycocotools.coco import COCO

        dataset_name = dataset_name or self.name
        timer = Timer()
        is_shot = "shot" in dataset_name
        if is_shot:
            # load low-shot annotations from category-wise json files
            # these files can be found at
            # http://dl.yf.io/fs-det/datasets/cocosplit/
            fileids = {}
            split_dir = os.path.join(self.data_root, "coco/few_shot_split")
            if "seed" in dataset_name:
                shot = dataset_name.split('_')[-2].split("shot")[0]
                seed = int(dataset_name.split("_seed")[-1])
                split_dir = os.path.join(split_dir, "seed{}".format(seed))
            else:
                shot = dataset_name.split('_')[-1].split("shot")[0]
            for idx, cls in enumerate(self.meta["thing_classes"]):
                json_file = os.path.join(split_dir,
                                         "full_box_{}shot_{}_trainval.json".
                                         format(shot, cls))
                json_file = PathManager.get_local_path(json_file)
                with contextlib.redirect_stdout(io.StringIO()):
                    coco_api = COCO(json_file)
                img_ids = sorted(list(coco_api.imgs.keys()))
                imgs = coco_api.loadImgs(img_ids)
                anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
                fileids[idx] = list(zip(imgs, anns))
        else:
            json_file = PathManager.get_local_path(json_file)
            with contextlib.redirect_stdout(io.StringIO()):
                coco_api = COCO(json_file)
            img_ids = sorted(list(coco_api.imgs.keys()))
            imgs = coco_api.loadImgs(img_ids)
            anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
            img_anns = list(zip(imgs, anns))

        if timer.seconds() > 1:
            logger.info("Loading {} takes {:.2f} seconds.".format(
                json_file, timer.seconds()))
        id_map = self.meta["thing_dataset_id_to_contiguous_id"]

        dataset_dicts = []
        ann_keys = ['iscrowd', 'bbox', 'category_id'] \
            + (extra_annotation_keys or [])

        if is_shot:
            for _, fileids_ in fileids.items():
                dicts = []
                for (img_dict, anno_dict_list) in fileids_:
                    for anno in anno_dict_list:
                        record = {}
                        record["file_name"] = \
                            os.path.join(image_root, img_dict["file_name"])
                        record["height"] = img_dict["height"]
                        record["width"] = img_dict["width"]
                        image_id = record["image_id"] = img_dict["id"]
                        assert anno["image_id"] == image_id
                        assert anno.get("ignore", 0) == 0

                        obj = {k: anno[k] for k in ann_keys if k in anno}
                        obj["bbox_mode"] = BoxMode.XYWH_ABS
                        obj["category_id"] = id_map[obj["category_id"]]
                        record["annotations"] = [obj]
                        dicts.append(record)
                if len(dicts) > int(shot):
                    dicts = np.random.choice(dicts, int(shot), replace=False)
                dataset_dicts.extend(dicts)

                if self.merge_few_shot_splits:
                    dataset_dicts = self._merge_dataset_dicts(dataset_dicts)
        else:
            for (img_dict, anno_dict_list) in img_anns:
                record = {}
                record["file_name"] = \
                    os.path.join(image_root, img_dict["file_name"])
                record["height"] = img_dict["height"]
                record["width"] = img_dict["width"]
                image_id = record["image_id"] = img_dict["id"]

                objs = []
                ignore = False
                for anno in anno_dict_list:
                    assert anno["image_id"] == image_id
                    assert anno.get("ignore", 0) == 0
                    obj = {k: anno[k] for k in ann_keys if k in anno}
                    obj["bbox_mode"] = BoxMode.XYWH_ABS
                    if obj["category_id"] in id_map:
                        obj["category_id"] = id_map[obj["category_id"]]
                        objs.append(obj)
                    elif self.filter_novel_img:  # Filter this image out
                        ignore = True
                        break
                if ignore:
                    continue
                record["annotations"] = objs
                dataset_dicts.append(record)
        return self.maybe_repeat(dataset_dicts)

    def maybe_repeat(self, dicts):
        assert len(dicts) != 0
        if len(dicts) < 10000 and self.is_train:
            new_dicts = []
            num_repeat = 10000 // len(dicts) + 1
            for _ in range(num_repeat):
                new_dicts.extend(dicts)
            return new_dicts
        return dicts
