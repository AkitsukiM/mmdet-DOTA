_base_ = [
    '../_base_/datasets/coco_dota.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

pretrained='/home/ubuntu/Workspace/YuHongtian/mmdetection-master/work_dirs/Conformer_small_patch32.pth'
num_stages = 7
cross_attention_flag = [False, True, True, True, True, True, True,]
num_proposals = 100
model = dict(
    type='ProgressiveProposalHighlight',
    backbone=dict(
        type='ConformerDet',
        embed_dim=384,
        depth=12,
        patch_size=32,
        channel_ratio=4,
        num_heads=6,
        drop_path_rate=0.1,
        norm_eval=True,
        frozen_stages=0,
        out_indices=(4, 8, 11, 12),
        init_cfg=dict(type='Pretrained', checkpoint=pretrained),
        proposal_token_num=num_proposals
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 1024],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='PPHead',
        num_proposals=num_proposals,
        embed_dim=384,
        proposal_highlight_channel=256),
    roi_head=dict(
        type='PHHead',
        num_stages=num_stages,
        stage_loss_weights=[1] * num_stages,
        proposal_highlight_channel=256,
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='ACAHead',
                num_classes=15,
                num_ffn_fcs=2,
                num_heads=8,
                num_cls_fcs=1,
                num_reg_fcs=3,
                feedforward_channels=2048,
                in_channels=256,
                dropout=0.0,
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                cross_attention_cfg=dict(
                    type='AugmentedCrossAttention',
                    dim=256,
                    num_heads=8,
                    mlp_ratio=4., 
                    qkv_bias=True, 
                    qk_scale=None, 
                    drop=0., 
                    attn_drop=0., 
                    drop_path=0., 
                    augmentation=64,
                    input_feat_shape=14),
                loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0),
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    clip_border=False,
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.5, 0.5, 1.0, 1.0]),
                cross_attention=flag
            ) for flag in cross_attention_flag
        ]),
    train_cfg=dict(
        rpn=None,
        rcnn=[
            dict(
                assigner=dict(
                    type='HungarianAssigner',
                    cls_cost=dict(type='FocalLossCost', weight=2.0),
                    reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                    iou_cost=dict(type='IoUCost', iou_mode='giou',
                                  weight=2.0)),
                sampler=dict(type='PseudoSampler'),
                pos_weight=1) for _ in range(num_stages) 
        ]),
    test_cfg=dict(rpn=None, rcnn=dict(max_per_img=num_proposals))
)


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
         type='AutoAugment',
         policies=[[
            dict(
                type='Resize',
                img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                           (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                           (736, 1333), (768, 1333), (800, 1333)],
                multiscale_mode='value',
                keep_ratio=True)
        ],
                  [
                      dict(
                          type='Resize',
                          img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                          multiscale_mode='value',
                          keep_ratio=True),
                      dict(
                          type='RandomCrop',
                          crop_type='absolute_range',
                          crop_size=(384, 600),
                          allow_negative_crop=True),
                      dict(
                          type='Resize',
                          img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                     (576, 1333), (608, 1333), (640, 1333),
                                     (672, 1333), (704, 1333), (736, 1333),
                                     (768, 1333), (800, 1333)],
                          multiscale_mode='value',
                          override=True,
                          keep_ratio=True)
                  ]]),
  dict(type='Normalize', **img_norm_cfg),
  dict(type='Pad', size_divisor=32),
  dict(type='DefaultFormatBundle'),
  dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
data = dict(
    samples_per_gpu=1, # 12GB GPU
    workers_per_gpu=2,
    train=dict(pipeline=train_pipeline)
)

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=1.25e-5,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'ln': dict(decay_mult=0.),
            'bn': dict(decay_mult=0.),
            'proposal_token': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=1, norm_type=2))
lr_config = dict(warmup_iters=1000, step=[27, 33])
runner = dict(max_epochs=36)
evaluation = dict(metric=['bbox'])
# add `find_unused_parameters=True` to avoid the error that the params not used in the detection
find_unused_parameters=True
