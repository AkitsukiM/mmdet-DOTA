import mmcv
import torch
import torch.nn as nn
from mmcv import tensor2imgs

from mmdet.models.builder import HEADS
from ...core import bbox_cxcywh_to_xyxy

from mmcv.cnn import normal_init

@HEADS.register_module()
class PPHead(nn.Module):

    def __init__(self,
                 num_proposals=100,
                 embed_dim=384,
                 proposal_highlight_channel=256,
                 **kwargs):
        super(PPHead, self).__init__()
        self.num_proposals = num_proposals
        self.proposal_highlight_channel = proposal_highlight_channel
        self.embed_dim = embed_dim
        self._init_layers()

    def _init_layers(self):
        self.init_proposal_bboxes = nn.Embedding(self.num_proposals, 4)
        self.init_proposal_features = nn.Embedding(
            self.num_proposals, self.proposal_highlight_channel)
        self.linear = nn.Linear(self.embed_dim, self.proposal_highlight_channel)

    def init_weights(self):
        nn.init.constant_(self.init_proposal_bboxes.weight[:, :2], 0.5)
        nn.init.constant_(self.init_proposal_bboxes.weight[:, 2:], 1)
        normal_init(self.linear, std=0.01)

    def _decode_init_proposals(self, imgs, img_metas, proposal_tokens):
        proposals = self.init_proposal_bboxes.weight.clone().detach()
        proposals = bbox_cxcywh_to_xyxy(proposals)
        num_imgs = len(imgs[0])
        imgs_whwh = []
        for meta in img_metas:
            h, w, _ = meta['img_shape']
            imgs_whwh.append(imgs[0].new_tensor([[w, h, w, h]]))
        imgs_whwh = torch.cat(imgs_whwh, dim=0)
        imgs_whwh = imgs_whwh[:, None, :]
        proposals = proposals * imgs_whwh

        proposal_tokens = self.linear(proposal_tokens)
        
        return proposals, proposal_tokens, imgs_whwh

    def forward_train(self, img, img_metas, proposal_tokens):
        return self._decode_init_proposals(img, img_metas, proposal_tokens)

    def simple_test_rpn(self, img, img_metas, proposal_tokens):
        return self._decode_init_proposals(img, img_metas, proposal_tokens)