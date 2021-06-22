import torch
import torch.nn as nn

from cvpods.layers import ShapeSpec
from cvpods.modeling.poolers import ROIPooler
from cvpods.modeling.roi_heads import StandardROIHeads
from cvpods.structures import Instances

from .fast_rcnn import CustomFastRCNNOutputs, GeneralizedFastRCNN
from .utils import class_split


class RedetectROIHeads(StandardROIHeads):
    def __init__(self, cfg, input_shape):
        super(StandardROIHeads, self).__init__(cfg, input_shape)
        self._init_box_head(cfg)
        self.basedet_bonus = cfg.MODEL.ROI_HEADS.BASEDET_BONUS

    def _init_box_head(self, cfg):
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / self.feature_strides[k]
                              for k in self.in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        self.train_on_pred_boxes = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        self.cls_agnostic_bbox_reg = \
            cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG
        self.consistency_coeff = cfg.MODEL.ROI_HEADS.CONSISTENCY_COEFF

        in_channels = [self.feature_channels[f] for f in self.in_features]
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.box_head = cfg.build_box_head(
            cfg, ShapeSpec(
                channels=in_channels,
                height=pooler_resolution,
                width=pooler_resolution
            )
        )

        input_size = self.box_head.output_size
        cosine_on = cfg.MODEL.ROI_HEADS.COSINE_ON
        cfg.MODEL.ROI_HEADS.COSINE_ON = False
        self.box_predictor = GeneralizedFastRCNN(cfg, input_size,
                                                 class_split.num_classes(
                                                     "base"),
                                                 self.cls_agnostic_bbox_reg)
        cfg.MODEL.ROI_HEADS.COSINE_ON = cosine_on
        self.redetector = GeneralizedFastRCNN(cfg, input_size,
                                              class_split.num_classes(),
                                              self.cls_agnostic_bbox_reg)

    def _forward_base_box(self, features, proposals):
        box_features = self.box_pooler(features,
                                       [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        pred_class_logits, pred_proposal_deltas = \
            self.box_predictor(box_features)
        return box_features, pred_class_logits, pred_proposal_deltas

    def _forward_novel_box(self, box_features, base_class_logits):
        probs = torch.softmax(base_class_logits, dim=1)
        pickup_indices = probs[:, :-1].max(dim=1)[0] <= self.confidence_thresh
        box_features = box_features[pickup_indices]
        pred_class_logits, pred_proposal_deltas = self.redetector(box_features)
        return pickup_indices, pred_class_logits, pred_proposal_deltas

    def forward(self, images, features, proposals, targets=None):
        if not self.training:
            return self.inference(images, features, proposals)
        del images
        assert targets
        proposals = self.label_and_sample_proposals(proposals, targets)
        del targets
        features_list = [features[f] for f in self.in_features]
        losses = self._forward_box(features_list, proposals)
        return proposals, losses

    def _forward_box(self, features, proposals):
        box_features = self.box_pooler(features,
                                       [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        novel_class_logits, novel_proposal_deltas = \
            self.redetector(box_features)
        outputs = CustomFastRCNNOutputs(
            self.box2box_transform,
            novel_class_logits,
            novel_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )
        losses = outputs.losses()

        base_class_logits = self.box_predictor(box_features, True)
        losses["loss_con_cls"] = self._cls_consistency_loss(
            base_class_logits, novel_class_logits)
        return losses

    def inference(self, images, features, proposals, targets=None):
        del images
        assert not self.training, "re-detect only supports inference"
        del targets
        features_list = [features[f] for f in self.in_features]

        box_features = self.box_pooler(features_list,
                                       [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        base_class_logits, base_proposal_deltas = \
            self.box_predictor(box_features)
        novel_class_logits, novel_proposal_deltas = \
            self.redetector(box_features)

        # resize base results
        temp_logits = base_class_logits.new_zeros(novel_class_logits.size())
        base_mapping = class_split.to_all_classes_bool(
            "base", device=temp_logits.device, include_bg=True)
        temp_logits[:, base_mapping] = base_class_logits[:, :-1]
        temp_logits[:, -1] = base_class_logits[:, -1]
        base_class_logits = temp_logits

        temp_deltas = base_proposal_deltas.new_zeros(
            novel_proposal_deltas.size())
        temp_deltas = temp_deltas.view(temp_deltas.size(0),
                                       class_split.num_classes(), 4)
        base_proposal_deltas = base_proposal_deltas.view(
            base_proposal_deltas.size(0), -1, 4)
        temp_deltas[:, base_mapping[:-1]] = base_proposal_deltas
        base_proposal_deltas = temp_deltas.view(temp_deltas.size(0), -1)

        assert base_class_logits.size(0) == novel_class_logits.size(0)
        final_logits = torch.cat([base_class_logits, novel_class_logits],
                                 dim=0)
        final_deltas = torch.cat([base_proposal_deltas, novel_proposal_deltas],
                                 dim=0)
        final_proposals = [Instances.cat([p, p]) for p in proposals]

        outputs = CustomFastRCNNOutputs(
            self.box2box_transform, final_logits, final_deltas,
            final_proposals, self.smooth_l1_beta
        )
        pred_instances, _ = outputs.inference(
            self.test_score_thresh, self.test_nms_thresh,
            self.test_nms_type, self.test_detections_per_img,
            self.basedet_bonus
        )
        return pred_instances, {}

    def _cls_consistency_loss(self, base_logits, novel_logits):
        base_mapping = class_split.to_all_classes_bool(
            "base", device=base_logits.device, include_bg=True)
        loss = nn.KLDivLoss(reduction="batchmean")
        novel_logits = novel_logits[:, base_mapping]
        base_logits = base_logits[:, :-1]
        base_log_probs = nn.LogSoftmax(dim=-1)(base_logits)
        novel_probs = torch.softmax(novel_logits, dim=-1)
        return loss(base_log_probs, novel_probs) * self.consistency_coeff
