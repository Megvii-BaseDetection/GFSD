from cvpods.layers import ShapeSpec
from cvpods.modeling.backbone import Backbone
from cvpods.modeling.backbone.fpn import build_resnet_fpn_backbone
from cvpods.modeling.proposal_generator import RPN
from cvpods.modeling.roi_heads.box_head import FastRCNNConvFCHead
from dataset import *  # noqa
from modeling import DoubleHeadRPN, RedetectROIHeads
from rcnn import RCNN


def build_backbone(cfg, input_shape=None):
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))
    backbone = build_resnet_fpn_backbone(cfg, input_shape)
    assert isinstance(backbone, Backbone)
    return backbone


def build_proposal_generator_train(cfg, input_shape):
    return RPN(cfg, input_shape)


def build_proposal_generator_inference(cfg, input_shape):
    return DoubleHeadRPN(cfg, input_shape)


def build_roi_heads(cfg, input_shape):
    return RedetectROIHeads(cfg, input_shape)


def build_box_head(cfg, input_shape):
    return FastRCNNConvFCHead(cfg, input_shape)


def build_model(cfg, training=True):
    cfg.build_backbone = build_backbone
    cfg.build_proposal_generator = build_proposal_generator_train \
        if training else build_proposal_generator_inference
    cfg.build_roi_heads = build_roi_heads
    cfg.build_box_head = build_box_head

    model = RCNN(cfg)
    return model
