_base_ = './cascade_rcnn_r50_fpn_1x_coco_dota.py'


# from './mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py' ##### #####


pretrained = '/home/marina/Workspace/Dataset/pretrained/swin_tiny_patch4_window7_224.pth'

model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[96, 192, 384, 768]))

img_norm_cfg = dict(
    # mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    mean=[81.93, 82.84, 78.56], std=[47.23, 45.73, 44.41], to_rgb=True)

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='BoxToBox', mode='rotated_box_to_bbox_np'),
    # dict(type='RandomFlip', flip_ratio=[0.25, 0.25], direction = ['horizontal', 'vertical']),
    # 
    # from https://github.com/csuhan/s2anet/blob/master/configs/hrsc2016/s2anet_r50_fpn_3x_hrsc2016.py ##### #####
    # 
    # dict(type='TempVisualize', note = '0-ori', img_rewrite = False, sys_exit = False),
    dict(type='RotatedRandomFlip', flip_ratio=0.5),
    # dict(type='TempVisualize', note = '1-flip', img_rewrite = False, sys_exit = False),
    dict(type='RandomRotate', rotate_ratio=0.5),
    # dict(type='TempVisualize', note = '2-rotate', img_rewrite = True, sys_exit = False),
    # 
    # From https://cocodataset.org/#format-data ,
    # we can know the original coco bbox format is:
    # [x, y, width, height]
    # and after {def _parse_ann_info in class CocoDataset in mmdet/datasets/coco.py} with {class LoadAnnotations in mmdet/datasets/pipelines/loading.py}
    # we can know the bbox format is:
    # [xmin, xmax, ymin, ymax]
    # 
    # However, when we create dota2coco json, we use the bbox format as
    # [x_ctr, y_ctr, width, height, angle]
    # To satisfy this format, we change the {def _parse_ann_info in class Dota2CocoDataset in mmdet/datasets/dota2coco.py}
    # so that the bbox entering 'RotatedRandomFlip' still has the format of [x_ctr, y_ctr, width, height, angle]
    # for both gt_bboxes and gt_bboxes_ignore
    # 
    # Finally, we use 'rotated_box_to_bbox_np' to change the bbox format to
    # [xmin, xmax, ymin, ymax]
    # so that the bbox entering 'Resize' would be correct
    # 
    dict(type='BoxToBox', mode='rotated_box_to_bbox_np'),
    # dict(type='TempVisualize', note='3-final', img_rewrite = False, sys_exit = True),
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

data_root = '/home/marina/Workspace/Dataset/DOTA-ImgSplit-COCO/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
# lr_config = dict(warmup_iters=1000, step=[27, 33])
lr_config = dict(_delete_=True, policy='CosineAnnealing', warmup='linear', warmup_iters=1000, warmup_ratio=0.001, min_lr=0.)

runner = dict(max_epochs=36)


# from './mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco.py' ##### #####


# you need to set mode='dynamic' if you are using pytorch<=1.5.0
fp16 = dict(loss_scale=dict(init_scale=512))

