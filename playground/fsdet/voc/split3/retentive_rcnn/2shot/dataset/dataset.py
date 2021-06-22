import logging
import os.path as osp
import xml.etree.ElementTree as ET

import numpy as np

from cvpods.data.datasets.voc import VOCDataset
from cvpods.data.detection_utils import create_keypoint_hflip_indices
from cvpods.data.registry import DATASETS
from cvpods.structures import BoxMode
from cvpods.utils import PathManager

from .paths_route import _PREDEFINED_SPLITS_VOC_FEWSHOT

logger = logging.getLogger(__name__)


@DATASETS.register()
class VOCFewShotDataset(VOCDataset):
    def __init__(self, cfg, dataset_name, transforms=[], is_train=True):
        super(VOCDataset, self).__init__(
            cfg, dataset_name, transforms, is_train
        )
        voc_fewshot_info = _PREDEFINED_SPLITS_VOC_FEWSHOT

        image_root, split = voc_fewshot_info["voc"][self.name]
        self.image_root = osp.join(self.data_root, image_root) \
            if "://" not in image_root else image_root
        self.split = split

        # Register few-shot attributes
        few_shot_keywords = ["all", "base", "novel"]
        self.keepclasses = [n for n in few_shot_keywords
                            if n in self.name]
        assert len(self.keepclasses) == 1, \
            "{} contains multiple or no keywords in {}".format(
                self.name, few_shot_keywords)
        self.keepclasses = self.keepclasses[0]
        self.sid = int(self.name.split(self.keepclasses)[1][0])

        self.meta = self._get_metadata()
        self.dataset_dicts = self._load_annotations()
        self._set_group_flag()

        self.eval_with_gt = cfg.TEST.get("WITH_GT", False)

        # fmt: off
        self.data_format = cfg.INPUT.FORMAT
        self.mask_on = cfg.MODEL.MASK_ON
        self.mask_format = cfg.INPUT.MASK_FORMAT
        self.keypoint_on = cfg.MODEL.KEYPOINT_ON
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS
        # fmt: on

        if self.keypoint_on:
            # Flip only makes sense in training
            self.keypoint_hflip_indices = \
                create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)
        else:
            self.keypoint_hflip_indices = None

    def _get_metadata(self):
        thing_classes = eval("PASCAL_VOC_{}_CATEGORIES".format(
            self.keepclasses.upper()))[self.sid]
        meta = {
            "thing_classes": thing_classes,
            "evaluator_type":
                _PREDEFINED_SPLITS_VOC_FEWSHOT["evaluator_type"]["voc"],
            "dirname": self.image_root,
            "split": self.split,
            "year": int(self.name.split('_')[1]),
        }
        return meta

    def _load_annotations(self):
        is_shots = "shot" in self.name
        if is_shots:
            fileids = {}
            print(self.image_root, self.data_root)
            split_dir = osp.join(self.data_root, "voc/few_shot_split")
            if "seed" in self.name:
                shot = self.name.split('_')[-2].split("shot")[0]
                seed = int(self.name.split("_seed")[-1])
                split_dir = osp.join(split_dir, "seed{}".format(seed))
            else:
                shot = self.name.split('_')[-1].split("shot")[0]
            for cls in self.meta["thing_classes"]:
                with PathManager.open(
                        osp.join(split_dir,
                                 "box_{}shot_{}_train.txt".format(
                                     shot, cls))
                ) as f:
                    fileids_ = np.loadtxt(f, dtype=np.str).tolist()
                    if isinstance(fileids_, str):
                        fileids_ = [fileids_]
                    fileids_ = [fid.split('/')[-1].split('.jpg')[0]
                                for fid in fileids_]
                    fileids[cls] = fileids_
        else:
            with PathManager.open(osp.join(
                    self.image_root, "ImageSets", "Main",
                    self.split + ".txt")) as f:
                fileids = np.loadtxt(f, dtype=np.str)

        dicts = []
        if is_shots:
            for cls, fileids_ in fileids.items():
                dicts_ = []
                for fid in fileids_:
                    year = "2012" if "_" in fid else "2007"
                    dirname = osp.join(self.data_root, "voc",
                                       "VOC{}".format(year))
                    anno_file = osp.join(dirname, "Annotations",
                                         fid + ".xml")
                    jpeg_file = osp.join(dirname, "JPEGImages",
                                         fid + ".jpg")

                    tree = ET.parse(anno_file)
                    for obj in tree.findall("object"):
                        r = {
                            "file_name": jpeg_file,
                            "image_id": fid,
                            "height":
                                int(tree.findall(
                                    "./size/height")[0].text),
                            "width":
                                int(tree.findall(
                                    "./size/width")[0].text),
                        }
                        cls_ = obj.find("name").text
                        if cls != cls_:
                            continue
                        bbox = obj.find("bndbox")
                        bbox = [float(bbox.find(x).text)
                                for x in ["xmin", "ymin", "xmax", "ymax"]]
                        bbox[0] -= 1.0
                        bbox[1] -= 1.0

                        instances = [{
                            "category_id":
                                self.meta["thing_classes"].index(cls),
                            "bbox": bbox,
                            "bbox_mode": BoxMode.XYXY_ABS
                        }]
                        r["annotations"] = instances
                        dicts_.append(r)
                if len(dicts_) > int(shot):
                    dicts_ = np.random.choice(dicts_,
                                              int(shot),
                                              replace=False)
                dicts.extend(dicts_)
        else:
            for fid in fileids:
                anno_file = osp.join(self.image_root,
                                     "Annotations", fid + ".xml")
                jpeg_file = osp.join(self.image_root,
                                     "JPEGImages", fid + ".jpg")

                tree = ET.parse(anno_file)
                r = {
                    "file_name": jpeg_file,
                    "image_id": fid,
                    "height":
                        int(tree.findall("./size/height")[0].text),
                    "width":
                        int(tree.findall("./size/width")[0].text),
                }
                ignore = False
                instances = []
                for obj in tree.findall("object"):
                    cls = obj.find("name").text
                    if not (cls in self.meta["thing_classes"]):
                        ignore = True
                        break

                    bbox = obj.find("bndbox")
                    bbox = [float(bbox.find(x).text)
                            for x in ["xmin", "ymin", "xmax", "ymax"]]
                    bbox[0] -= 1.0
                    bbox[1] -= 1.0

                    instances.append({
                        "category_id":
                            self.meta["thing_classes"].index(cls),
                        "bbox": bbox,
                        "bbox_mode": BoxMode.XYXY_ABS,
                    })
                if ignore:
                    continue
                r["annotations"] = instances
                dicts.append(r)
        return dicts


# fmt: off
CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

# PASCAL VOC few-shot benchmark splits
PASCAL_VOC_ALL_CATEGORIES = {
    1: ['aeroplane', 'bicycle', 'boat', 'bottle', 'car', 'cat', 'chair',
        'diningtable', 'dog', 'horse', 'person', 'pottedplant', 'sheep',
        'train', 'tvmonitor', 'bird', 'bus', 'cow', 'motorbike', 'sofa'],
    2: ['bicycle', 'bird', 'boat', 'bus', 'car', 'cat', 'chair', 'diningtable',
        'dog', 'motorbike', 'person', 'pottedplant', 'sheep', 'train',
        'tvmonitor', 'aeroplane', 'bottle', 'cow', 'horse', 'sofa'],
    3: ['aeroplane', 'bicycle', 'bird', 'bottle', 'bus', 'car', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'person', 'pottedplant', 'train',
        'tvmonitor', 'boat', 'cat', 'motorbike', 'sheep', 'sofa'],
}

PASCAL_VOC_NOVEL_CATEGORIES = {
    1: ['bird', 'bus', 'cow', 'motorbike', 'sofa'],
    2: ['aeroplane', 'bottle', 'cow', 'horse', 'sofa'],
    3: ['boat', 'cat', 'motorbike', 'sheep', 'sofa'],
}

PASCAL_VOC_BASE_CATEGORIES = {
    1: ['aeroplane', 'bicycle', 'boat', 'bottle', 'car', 'cat', 'chair',
        'diningtable', 'dog', 'horse', 'person', 'pottedplant', 'sheep',
        'train', 'tvmonitor'],
    2: ['bicycle', 'bird', 'boat', 'bus', 'car', 'cat', 'chair', 'diningtable',
        'dog', 'motorbike', 'person', 'pottedplant', 'sheep', 'train',
        'tvmonitor'],
    3: ['aeroplane', 'bicycle', 'bird', 'bottle', 'bus', 'car', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'person', 'pottedplant', 'train',
        'tvmonitor'],
}
