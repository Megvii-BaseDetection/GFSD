from cvpods.data.datasets.builtin_meta import (COCO_CATEGORIES,
                                                _get_coco_instances_meta)

# Novel COCO categories
COCO_NOVEL_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "person"},
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "bicycle"},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "car"},
    {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "motorcycle"},
    {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": "airplane"},
    {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": "bus"},
    {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": "train"},
    {"color": [0, 0, 192], "isthing": 1, "id": 9, "name": "boat"},
    {"color": [165, 42, 42], "isthing": 1, "id": 16, "name": "bird"},
    {"color": [255, 77, 255], "isthing": 1, "id": 17, "name": "cat"},
    {"color": [0, 226, 252], "isthing": 1, "id": 18, "name": "dog"},
    {"color": [182, 182, 255], "isthing": 1, "id": 19, "name": "horse"},
    {"color": [0, 82, 0], "isthing": 1, "id": 20, "name": "sheep"},
    {"color": [120, 166, 157], "isthing": 1, "id": 21, "name": "cow"},
    {"color": [197, 226, 255], "isthing": 1, "id": 44, "name": "bottle"},
    {"color": [153, 69, 1], "isthing": 1, "id": 62, "name": "chair"},
    {"color": [3, 95, 161], "isthing": 1, "id": 63, "name": "couch"},
    {"color": [163, 255, 0], "isthing": 1, "id": 64, "name": "potted plant"},
    {"color": [0, 182, 199], "isthing": 1, "id": 67, "name": "dining table"},
    {"color": [183, 130, 88], "isthing": 1, "id": 72, "name": "tv"},
]


def _get_coco_fewshot_instances_meta():
    ret = _get_coco_instances_meta()
    novel_ids = [k["id"] for k in COCO_NOVEL_CATEGORIES if k["isthing"] == 1]
    novel_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(novel_ids)}
    novel_classes = [k["name"] for k in
                     COCO_NOVEL_CATEGORIES if k["isthing"] == 1]
    novel_colors = [k["color"] for k in
                    COCO_NOVEL_CATEGORIES if k["isthing"] == 1]
    base_categories = [k for k in COCO_CATEGORIES
                       if k["isthing"] == 1 and k["name"] not in novel_classes]
    base_ids = [k["id"] for k in base_categories]
    base_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(base_ids)}
    base_classes = [k["name"] for k in base_categories]
    base_colors = [k["color"] for k in base_categories]
    ret["novel_dataset_id_to_contiguous_id"] = \
        novel_dataset_id_to_contiguous_id
    ret["novel_classes"] = novel_classes
    ret["novel_colors"] = novel_colors
    ret["base_dataset_id_to_contiguous_id"] = base_dataset_id_to_contiguous_id
    ret["base_classes"] = base_classes
    ret["base_colors"] = base_colors
    return ret
