_base_ = './cascade_rcnn_swin-b-p4-w7_fpn_fp16_ms-crop-3x_coco_dota.py'


model = dict(
    test_cfg=dict(
        rcnn=dict(
            max_per_img=1000)))

img_norm_cfg = dict(
    # mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    mean=[81.93, 82.84, 78.56], std=[47.23, 45.73, 44.41], to_rgb=True)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # from https://github.com/sfzhang15/ATSS/blob/master/configs/atss/atss_dcnv2_X_101_64x4d_FPN_2x.yaml ##### #####
        img_scale=[(400, 3000), (500, 3000), (600, 3000), (640, 3000),
                   (700, 3000), (900, 3000), (1000, 3000), (1100, 3000),
                   (1200, 3000), (1300, 3000), (1400, 3000), (1800, 3000)],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data_root = '/home/marina/Workspace/Dataset/DOTA-ImgSplit-COCO/'
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    test=dict(
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))

