import os.path as osp

from cvpods.configs.rcnn_fpn_config import RCNNFPNConfig

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="/data/fsdet_models/voc/split1/base_model.pth",
        MASK_ON=False,
        RESNETS=dict(DEPTH=101),
        ROI_HEADS=dict(
            NUM_CLASSES=20,
            COSINE_ON=True,
            COSINE_SCALE=20,
            BOX_REG_ON=True,
            FREEZE_FEAT=True,
            BASEDET_BONUS=.1,
            CONSISTENCY_COEFF=0.1,
        ),
        BACKBONE=dict(FREEZE=True),
        PROPOSAL_GENERATOR=dict(FREEZE_FEAT=True, FREEZE_BOX=True),
    ),
    GLOBAL=dict(DUMP_TEST=True),
    DATASETS=dict(
        TRAIN=("vocfsdet_2007_trainval_all3_2shot",),
        TEST=("vocfsdet_2007_test_all3",),
    ),
    SOLVER=dict(
        IMS_PER_BATCH=16,
        OPTIMIZER=dict(
            BASE_LR=0.05,
        ),
        LR_SCHEDULER=dict(
            STEPS=(9000,),
            MAX_ITER=9000,
            WARMUP_ITER=100,
        ),
    ),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
                ("ResizeShortestEdge",
                 dict(short_edge_length=(480, 512, 544, 576, 608, 
                                         640, 672, 704, 736, 768, 800),
                      max_size=1333, sample_style="choice")),
                ("RandomFlip", dict()),
            ],
            TEST_PIPELINES=[
                ("ResizeShortestEdge",
                 dict(short_edge_length=800, max_size=1333, sample_style="choice")),
            ],
        )
    ),
    OUTPUT_DIR=osp.join(
        '/data/Outputs/model_logs/cvpods_playground',
        osp.split(osp.realpath(__file__))[0].split("playground/")[-1]),
)


class FasterRCNNConfig(RCNNFPNConfig):
    def __init__(self):
        super(FasterRCNNConfig, self).__init__()
        self._register_configuration(_config_dict)


config = FasterRCNNConfig()
