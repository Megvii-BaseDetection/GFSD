from cvpods.data.registry import PATH_ROUTES

_PREDEFINED_SPLITS_VOC_FEWSHOT = {}
_PREDEFINED_SPLITS_VOC_FEWSHOT["dataset_type"] = "VOCFewShotDataset"
_PREDEFINED_SPLITS_VOC_FEWSHOT["evaluator_type"] = {
    "voc": "pascal_voc",
}
_PREDEFINED_SPLITS_VOC_FEWSHOT["voc"] = {
    # these json files can be downloaded at
    # http://dl.yf.io/fs-det/datasets/cocosplit/datasplit/
    "voc_2007_trainval_base1": (
        "voc/VOC2007", "trainval"
    ),
    "vocfsdet_2007_trainval_base2": (
        "voc/VOC2007", "trainval"
    ),
    "vocfsdet_2007_trainval_base3": (
        "voc/VOC2007", "trainval"
    ),
    "vocfsdet_2012_trainval_base1": (
        "voc/VOC2012", "trainval"
    ),
    "vocfsdet_2012_trainval_base2": (
        "voc/VOC2012", "trainval"
    ),
    "vocfsdet_2012_trainval_base3": (
        "voc/VOC2012", "trainval"
    ),
    "vocfsdet_2007_trainval_all1": (
        "voc/VOC2007", "trainval"
    ),
    "vocfsdet_2007_trainval_all2": (
        "voc/VOC2007", "trainval"
    ),
    "vocfsdet_2007_trainval_all3": (
        "voc/VOC2007", "trainval"
    ),
    "vocfsdet_2012_trainval_all1": (
        "voc/VOC2012", "trainval"
    ),
    "vocfsdet_2012_trainval_all2": (
        "voc/VOC2012", "trainval"
    ),
    "vocfsdet_2012_trainval_all3": (
        "voc/VOC2012", "trainval"
    ),
    "vocfsdet_2007_test_base1": ("voc/VOC2007", "test"),
    "vocfsdet_2007_test_base2": ("voc/VOC2007", "test"),
    "vocfsdet_2007_test_base3": ("voc/VOC2007", "test"),
    "vocfsdet_2007_test_novel1": ("voc/VOC2007", "test"),
    "vocfsdet_2007_test_novel2": ("voc/VOC2007", "test"),
    "vocfsdet_2007_test_novel3": ("voc/VOC2007", "test"),
    "vocfsdet_2007_test_all1": ("voc/VOC2007", "test"),
    "vocfsdet_2007_test_all2": ("voc/VOC2007", "test"),
    "vocfsdet_2007_test_all3": ("voc/VOC2007", "test"),
}


for prefix in ["all", "novel"]:
    for sid in [1, 2, 3]:
        for shot in [1, 2, 3, 5, 10]:
            for year in [2007, 2012]:
                for seed in range(30):
                    seed = "" if seed == 0 else f"_seed{seed}"
                    name = f"vocfsdet_{year}_trainval" \
                           f"_{prefix}{sid}_{shot}shot{seed}"
                    dirname = f"voc/VOC{year}"
                    img_file = f"{prefix}_{shot}shot" \
                               f"_split_{sid}_trainval"
                    _PREDEFINED_SPLITS_VOC_FEWSHOT["voc"][name] = \
                        (dirname, img_file)

PATH_ROUTES.register(_PREDEFINED_SPLITS_VOC_FEWSHOT, "VOCFSDET")
