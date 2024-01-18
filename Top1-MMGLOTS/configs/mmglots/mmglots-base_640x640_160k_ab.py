_base_ = [
    '../_base_/models/mmglots.py', '../_base_/datasets/c2seg_ab.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_640k.py'
]

# model cfg
model = dict(
    pretrained='pretrain/beitv2_base_patch16_224_pt1k_ft21k.pth',
    backbone=dict(hsi_channels=242),
    decode_head=dict(num_classes=13),
    auxiliary_head=dict(num_classes=13),
    test_cfg=dict(mode='slide', crop_size=(640, 640), stride=(426, 426)))

# optimizer cfg
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=3e-5,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    # constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.9))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)


data = dict(
    train=dict(
        img_dir='img_dir/train',
        ann_dir='ann_dir/train'),
    samples_per_gpu=1)
