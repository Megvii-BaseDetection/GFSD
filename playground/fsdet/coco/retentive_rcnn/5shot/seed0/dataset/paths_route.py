from cvpods.data.registry import PATH_ROUTES

_PREDEFINED_SPLITS_COCO_FEWSHOT = {}
_PREDEFINED_SPLITS_COCO_FEWSHOT["dataset_type"] = "COCOFewShotDataset"
_PREDEFINED_SPLITS_COCO_FEWSHOT["evaluator_type"] = {
    "coco": "coco",
}
_PREDEFINED_SPLITS_COCO_FEWSHOT["coco"] = {
    # these json files can be downloaded at
    # http://dl.yf.io/fs-det/datasets/cocosplit/datasplit/
    "cocofsdet_2014_trainval_all": (
        "coco/trainval2014",
        "coco/few_shot_split/datasplit/trainvalno5k.json"
    ),
    "cocofsdet_2014_trainval_base": (
        "coco/trainval2014",
        "coco/few_shot_split/datasplit/trainvalno5k.json"
    ),
    "cocofsdet_2014_test_all": ("coco/val2014",
                                "coco/few_shot_split/datasplit/5k.json"),
    "cocofsdet_2014_test_base": ("coco/val2014",
                                 "coco/few_shot_split/datasplit/5k.json"),
    "cocofsdet_2014_test_novel": ("coco/val2014",
                                  "coco/few_shot_split/datasplit/5k.json"),
}

for prefix in ["all", "novel"]:
    for shot in [1, 2, 3, 5, 10, 30]:
        for seed in range(10):
            seed = "" if seed == 0 else f"_seed{seed}"
            name = f"cocofsdet_2014_trainval_{prefix}_{shot}shot{seed}"
            _PREDEFINED_SPLITS_COCO_FEWSHOT["coco"][name] = \
                ("coco/trainval2014", "")

PATH_ROUTES.register(_PREDEFINED_SPLITS_COCO_FEWSHOT, "COCOFSDET")
