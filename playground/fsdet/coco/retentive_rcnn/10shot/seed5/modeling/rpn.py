import torch
import torch.nn as nn
import torch.nn.functional as F

from cvpods.modeling.anchor_generator import DefaultAnchorGenerator
from cvpods.modeling.box_regression import Box2BoxTransform
from cvpods.modeling.matcher import Matcher
from cvpods.modeling.proposal_generator.rpn import RPN, StandardRPNHead


class RPNHead(StandardRPNHead):
    def __init__(self, cfg, input_shape):
        super(StandardRPNHead, self).__init__()

        # Standard RPN is shared across levels:
        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, \
            "Each level must have the same channel!"
        in_channels = in_channels[0]

        # RPNHead should take the same input as anchor generator
        anchor_generator = DefaultAnchorGenerator(cfg, input_shape)
        num_cell_anchors = anchor_generator.num_cell_anchors
        box_dim = anchor_generator.box_dim
        assert (
            len(set(num_cell_anchors)) == 1
        ), "Each level must have the same number of cell anchors"
        num_cell_anchors = num_cell_anchors[0]

        # 3x3 conv for the hidden representation
        self.conv = nn.Conv2d(in_channels, in_channels,
                              kernel_size=3, stride=1, padding=1)
        # 1x1 conv for predicting objectness logits
        self.objectness_logits = nn.Conv2d(in_channels, num_cell_anchors,
                                           kernel_size=1, stride=1, bias=True)
        self.finetuned_objectness_logits = nn.Conv2d(
            in_channels, num_cell_anchors, kernel_size=1, stride=1, bias=True
        )
        # 1x1 conv for predicting box2box transform deltas
        self.anchor_deltas = nn.Conv2d(
            in_channels, num_cell_anchors * box_dim, kernel_size=1, stride=1
        )

        for layer in [self.conv, self.objectness_logits, self.anchor_deltas]:
            nn.init.normal_(layer.weight, std=0.01)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def forward(self, features):
        pred_objectness_logits = []
        pred_anchor_deltas = []
        for x in features:
            t = F.relu(self.conv(x))
            pred_anchor_deltas.append(self.anchor_deltas(t))
            logit_map = torch.max(self.objectness_logits(t),
                                  self.finetuned_objectness_logits(t))
            pred_objectness_logits.append(logit_map)
        return pred_objectness_logits, pred_anchor_deltas


class DoubleHeadRPN(RPN):
    def __init__(self, cfg, input_shape):
        super(RPN, self).__init__()

        # fmt: off
        self.min_box_side_len = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
        self.in_features = cfg.MODEL.RPN.IN_FEATURES
        self.nms_thresh = cfg.MODEL.RPN.NMS_THRESH
        self.nms_type = cfg.MODEL.RPN.NMS_TYPE
        self.batch_size_per_image = cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE
        self.positive_fraction = cfg.MODEL.RPN.POSITIVE_FRACTION
        self.smooth_l1_beta = cfg.MODEL.RPN.SMOOTH_L1_BETA
        self.loss_weight = cfg.MODEL.RPN.LOSS_WEIGHT
        # fmt: on

        # Map from self.training state to train/test settings
        self.pre_nms_topk = {
            True: cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN,
            False: cfg.MODEL.RPN.PRE_NMS_TOPK_TEST,
        }
        self.post_nms_topk = {
            True: cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN,
            False: cfg.MODEL.RPN.POST_NMS_TOPK_TEST,
        }
        self.boundary_threshold = cfg.MODEL.RPN.BOUNDARY_THRESH

        self.anchor_generator = DefaultAnchorGenerator(
            cfg, [input_shape[f] for f in self.in_features]
        )
        self.box2box_transform = Box2BoxTransform(
            weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS
        )
        self.anchor_matcher = Matcher(
            cfg.MODEL.RPN.IOU_THRESHOLDS, cfg.MODEL.RPN.IOU_LABELS,
            allow_low_quality_matches=True
        )
        self.rpn_head = RPNHead(cfg,
                                [input_shape[f] for f in self.in_features])
