import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cvpods.modeling.roi_heads.fast_rcnn import (FastRCNNOutputLayers,
                                                  FastRCNNOutputs,
                                                  fast_rcnn_inference)


class CustomFastRCNNOutputs(FastRCNNOutputs):
    def predict_probs(self, score_thresh=0.05, base_bonus=0.):
        probs = F.softmax(self.pred_class_logits, dim=-1)
        bonus = probs.new_zeros(probs.size())
        bonus[probs > score_thresh] = base_bonus
        # only the first half comes from base detector
        num_basedet = probs.size(0) // 2
        bonus[num_basedet:, :] = 0
        probs += bonus
        return probs.split(self.num_preds_per_image, dim=0)

    def inference(self, score_thresh, nms_thresh, nms_type,
                  topk_per_image, bonus=0.):
        boxes = self.predict_boxes()
        scores = self.predict_probs(score_thresh, bonus)
        image_shapes = self.image_shapes

        return fast_rcnn_inference(
            boxes, scores, image_shapes, score_thresh,
            nms_thresh, nms_type, topk_per_image
        )


class GeneralizedFastRCNN(FastRCNNOutputLayers):
    def __init__(self, cfg, input_size, num_classes, cls_agnostic_bbox_reg,
                 box_dim=4):
        """
        if box_dim is 0, bbox reg is not enabled
        """
        super(FastRCNNOutputLayers, self).__init__()
        if not hasattr(cfg.MODEL.ROI_HEADS, "COSINE_ON"):
            logging.getLogger(__name__).info(
                "cosine_on not given, using fc classifier by default")
            self.cosine = False
        else:
            self.cosine = cfg.MODEL.ROI_HEADS.COSINE_ON

        if not isinstance(input_size, int):
            input_size = np.prod(input_size)

        self.num_classes = num_classes
        self.cls_score = nn.Linear(
            input_size, num_classes + 1, bias=not self.cosine
        )
        nn.init.normal_(self.cls_score.weight, std=0.01)
        if not self.cosine:
            nn.init.constant_(self.cls_score.bias, 0)
        else:
            assert hasattr(cfg.MODEL.ROI_HEADS, "COSINE_SCALE")
            self.scale = cfg.MODEL.ROI_HEADS.COSINE_SCALE

        if box_dim:
            num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
            self.bbox_pred = nn.Linear(
                input_size, num_bbox_reg_classes * box_dim
            )
            nn.init.normal_(self.bbox_pred.weight, std=0.001)
            nn.init.constant_(self.bbox_pred.bias, 0)
        else:
            self.bbox_pred = None

    def forward(self, x, cls_only=False):
        if x.size(0) == 0:
            device = x.device
            score_size = self.num_classes + 1
            delta_size = 4
            return torch.zeros(0, score_size, device=device), \
                torch.zeros(0, delta_size, device=device)

        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)

        if self.cosine:
            # normalize the input x along the `input_size` dimension
            x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
            x_normalized = x.div(x_norm + 1e-5)

            # normalize weight
            temp_norm = torch.norm(self.cls_score.weight.data,
                                   p=2, dim=1).unsqueeze(1).expand_as(
                self.cls_score.weight.data)
            self.cls_score.weight.data = self.cls_score.weight.data.div(
                temp_norm + 1e-5)
            cos_dist = self.cls_score(x_normalized)
            scores = self.scale * cos_dist
        else:
            scores = self.cls_score(x)
        if cls_only:
            return scores

        if self.bbox_pred:
            proposal_deltas = self.bbox_pred(x)
        else:
            proposal_deltas = torch.zeros(scores.size(0), 4).to(scores.device)

        return scores, proposal_deltas
