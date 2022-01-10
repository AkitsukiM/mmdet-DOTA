import torch

from mmdet.core import bbox2result, bbox2roi, bbox_xyxy_to_cxcywh
from mmdet.core.bbox.samplers import PseudoSampler
from ..builder import HEADS
from .cascade_roi_head import CascadeRoIHead
# from timm.models.layers import trunc_normal_, DropPath
import torch.nn as nn
from mmcv.cnn import normal_init
from mmcv.cnn import Conv2d, ConvModule, build_upsample_layer
from mmcv.cnn import (bias_init_with_prob, build_activation_layer,
                      build_norm_layer)
import torch.nn.functional as F
from mmdet.models.builder import HEADS, build_loss

@HEADS.register_module()
class PHHead(CascadeRoIHead):
    def __init__(self,
                 num_stages=6,
                 stage_loss_weights=(1, 1, 1, 1, 1, 1),
                 proposal_highlight_channel=256,
                 bbox_roi_extractor=dict(
                     type='SingleRoIExtractor',
                     roi_layer=dict(
                         type='RoIAlign', output_size=7, sampling_ratio=2),
                     out_channels=256,
                     featmap_strides=[4, 8, 16, 32]),
                 bbox_head=dict(
                     type='DIIHead',
                     num_classes=80,
                     num_fcs=2,
                     num_heads=8,
                     num_cls_fcs=1,
                     num_reg_fcs=3,
                     feedforward_channels=2048,
                     hidden_channels=256,
                     dropout=0.0,
                     roi_feat_size=7,
                     ffn_act_cfg=dict(type='ReLU', inplace=True)),
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        assert len(stage_loss_weights) == num_stages
        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights
        self.proposal_highlight_channel = proposal_highlight_channel
        super(PHHead, self).__init__(
            num_stages,
            stage_loss_weights,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg)
        # train_cfg would be None when run the test.py
        if train_cfg is not None:
            for stage in range(num_stages):
                assert isinstance(self.bbox_sampler[stage], PseudoSampler), \
                    'Sparse R-CNN only support `PseudoSampler`'
    
    def _bbox_forward(self, stage, x, rois, object_feats, img_metas):
        num_imgs = len(img_metas)
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                        rois)
        cls_score, bbox_pred, object_feats = bbox_head(bbox_feats,
                                                       object_feats)
        proposal_list = self.bbox_head[stage].refine_bboxes(
            rois,
            rois.new_zeros(len(rois)),  # dummy arg
            bbox_pred.view(-1, bbox_pred.size(-1)),
            [rois.new_zeros(object_feats.size(1)) for _ in range(num_imgs)],
            img_metas)
        bbox_results = dict(
            cls_score=cls_score,
            decode_bbox_pred=torch.cat(proposal_list),
            object_feats=object_feats,
            # detach then use it in label assign
            detach_cls_score_list=[
                cls_score[i].detach() for i in range(num_imgs)
            ],
            detach_proposal_list=[item.detach() for item in proposal_list]
        )

        return bbox_results

    def forward_train(self,
                      x,
                      proposal_boxes,
                      proposal_features,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      imgs_whwh=None):
        num_imgs = len(img_metas)
        num_proposals = proposal_boxes.size(1)
        imgs_whwh = imgs_whwh.repeat(1, num_proposals, 1)
        all_stage_bbox_results = []
        proposal_list = [proposal_boxes[i] for i in range(len(proposal_boxes))]
        object_feats = proposal_features
            
        all_stage_loss = {}
        for stage in range(self.num_stages):
            rois = bbox2roi(proposal_list)
            bbox_results = self._bbox_forward(stage, x, rois, object_feats, img_metas)
            all_stage_bbox_results.append(bbox_results)
            if gt_bboxes_ignore is None:
                # TODO support ignore
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            cls_pred_list = bbox_results['detach_cls_score_list']
            proposal_list = bbox_results['detach_proposal_list']
            for i in range(num_imgs):
                normolize_bbox_ccwh = bbox_xyxy_to_cxcywh(proposal_list[i] /
                                                          imgs_whwh[i])
                assign_result = self.bbox_assigner[stage].assign(
                    normolize_bbox_ccwh, cls_pred_list[i], gt_bboxes[i],
                    gt_labels[i], img_metas[i])
                sampling_result = self.bbox_sampler[stage].sample(
                    assign_result, proposal_list[i], gt_bboxes[i])
                sampling_results.append(sampling_result)
            bbox_targets = self.bbox_head[stage].get_targets(
                sampling_results, gt_bboxes, gt_labels, self.train_cfg[stage],
                True)
            cls_score = bbox_results['cls_score']
            decode_bbox_pred = bbox_results['decode_bbox_pred']

            single_stage_loss = self.bbox_head[stage].loss(
                cls_score.view(-1, cls_score.size(-1)),
                decode_bbox_pred.view(-1, 4),
                *bbox_targets,
                imgs_whwh=imgs_whwh, stage=stage)
            for key, value in single_stage_loss.items():
                all_stage_loss[f'stage{stage}_{key}'] = value * \
                                    self.stage_loss_weights[stage]
            object_feats = bbox_results['object_feats']
        return all_stage_loss

    def simple_test(self,
                    x,
                    proposal_boxes,
                    proposal_features,
                    img_metas,
                    imgs_whwh,
                    rescale=False,
                    attns_maps=None):
        assert self.with_bbox, 'Bbox head must be implemented.'
        # Decode initial proposals
        num_imgs = len(img_metas)
        proposal_list = [proposal_boxes[i] for i in range(num_imgs)]
        object_feats = proposal_features
        num_proposals = object_feats.size(1)
    
        for stage in range(self.num_stages):
            rois = bbox2roi(proposal_list)                
            bbox_results = self._bbox_forward(stage, x, rois, object_feats,
                                              img_metas)
            object_feats = bbox_results['object_feats']
            cls_score = bbox_results['cls_score']
            proposal_list = bbox_results['detach_proposal_list']

        num_classes = self.bbox_head[-1].num_classes
        det_bboxes = []
        det_labels = []

        if self.bbox_head[-1].loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
        else:
            cls_score = cls_score.softmax(-1)[..., :-1]

        for img_id in range(num_imgs):
            cls_score_per_img = cls_score[img_id]
            scores_per_img, topk_indices = cls_score_per_img.flatten(
                0, 1).topk(
                    self.test_cfg.max_per_img, sorted=False)
            labels_per_img = topk_indices % num_classes
            bbox_pred_per_img = proposal_list[img_id][topk_indices //
                                                      num_classes]
            if rescale:
                scale_factor = img_metas[img_id]['scale_factor']
                bbox_pred_per_img /= bbox_pred_per_img.new_tensor(scale_factor)
            det_bboxes.append(
                torch.cat([bbox_pred_per_img, scores_per_img[:, None]], dim=1))
            det_labels.append(labels_per_img)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i], num_classes)
            for i in range(num_imgs)
        ]

        return bbox_results