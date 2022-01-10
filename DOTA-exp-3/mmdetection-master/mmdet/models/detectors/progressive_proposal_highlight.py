from ..builder import DETECTORS
from .two_stage import TwoStageDetector
from mmdet.core import (bbox2roi, bbox_mapping, merge_aug_bboxes, merge_aug_masks, multiclass_nms)
from mmdet.core import bbox2result, bbox2roi, bbox_xyxy_to_cxcywh

@DETECTORS.register_module()
class ProgressiveProposalHighlight(TwoStageDetector):
    def __init__(self, *args, **kwargs):
        super(ProgressiveProposalHighlight, self).__init__(*args, **kwargs)
        
    def extract_feat(self, img):
        x = self.backbone(img)
        xs, proposal_tokens = x[0], x[1]
        if self.with_neck:
            x = self.neck(xs)
        return x, proposal_tokens
    
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        
        x, proposal_tokens = self.extract_feat(img)
        proposal_boxes, proposal_tokens, imgs_whwh = \
            self.rpn_head.forward_train(x, img_metas, proposal_tokens)
        roi_losses = self.roi_head.forward_train(
            x,
            proposal_boxes,
            proposal_tokens,
            img_metas,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore=gt_bboxes_ignore,
            imgs_whwh=imgs_whwh)
        return roi_losses

    def simple_test(self, img, img_metas, rescale=False):
        x, proposal_tokens = self.extract_feat(img)
        proposal_boxes, proposal_tokens, imgs_whwh = \
            self.rpn_head.simple_test_rpn(x, img_metas, proposal_tokens)
        bbox_results = self.roi_head.simple_test(
            x,
            proposal_boxes,
            proposal_tokens,
            img_metas,
            imgs_whwh=imgs_whwh,
            rescale=rescale)
        return bbox_results