from cvpods.modeling.meta_arch import GeneralizedRCNN


class RCNN(GeneralizedRCNN):
    def __init__(self, cfg):
        super().__init__(cfg)

        # Freeze certain parameters
        if getattr(cfg.MODEL.BACKBONE, "FREEZE", False):
            print("backbone frozen")
            for p in self.backbone.parameters():
                p.requires_grad = False

        if getattr(cfg.MODEL.PROPOSAL_GENERATOR, "FREEZE", False):
            print("rpn frozen")
            for p in self.proposal_generator.parameters():
                p.requires_grad = False

        if getattr(cfg.MODEL.PROPOSAL_GENERATOR, "FREEZE_FEAT", False):
            print("rpn conv frozen")
            for p in self.proposal_generator.rpn_head.conv.parameters():
                p.requires_grad = False

        if getattr(cfg.MODEL.PROPOSAL_GENERATOR, "FREEZE_BOX", False):
            print("rpn bbox frozen")
            for p in self.proposal_generator. \
                    rpn_head.anchor_deltas.parameters():
                p.requires_grad = False

        if getattr(cfg.MODEL.PROPOSAL_GENERATOR, "FREEZE_CLS", False):
            print("rpn objectness frozen")
            for p in self.proposal_generator. \
                    rpn_head.objectness_logits.parameters():
                p.requires_grad = False

        if getattr(cfg.MODEL.ROI_HEADS, "FREEZE_FEAT", False):
            print("box head frozen")
            for p in self.roi_heads.box_head.parameters():
                p.requires_grad = False
            for p in self.roi_heads.box_predictor.parameters():
                p.requires_grad = False

    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)
        else:
            return super(RCNN, self).forward(batched_inputs)
