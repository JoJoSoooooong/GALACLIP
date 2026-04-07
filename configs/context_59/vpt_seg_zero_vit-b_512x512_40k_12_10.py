_base_ = [
    '../_base_/models/galaclip.py', '../_base_/datasets/context_59_512x512.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]

img_size = 512
in_channels = 512
out_indices = [11]

base_class =  [1,  4,  5,  6,  8, 
              10, 11, 12, 14, 15, 16, 18, 
              20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 
              30, 32, 33, 34, 35, 36, 38, 39, 
              40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 
              50, 51, 52, 53, 54, 55, 56, 57, 58]

novel_class = [0, 2, 3, 7, 9, 13, 17, 19, 31, 37]

both_class = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 
              10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 
              20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 
              30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 
              40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 
              50, 51, 52, 53, 54, 55, 56, 57, 58]

num_classes = len(base_class)

pretrained = '/hy-tmp/pretrained/ViT-B-16.pt'

model = dict(
    type='GalaCLIP',
    pretrained=pretrained, 
    pretrained_text=pretrained, 
    context_length=77,
    backbone=dict(
        type='VPTCLIPVisionTransformer',
        patch_size=16,
        width=768,
        output_dim=512,
        get_embeddings=True,
        drop_path_rate=0.1,
        layers=12,
        input_resolution=img_size,
        out_indices=out_indices,
        num_tokens=35,
        prompt_dim=768,
        total_d_layer=11,
        style='pytorch'),
    text_encoder=dict(
        type='CLIPTextEncoder',
        context_length=77,
        embed_dim=512,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        style='pytorch'),
    decode_head=dict(
        type='GalaFusionHead',
        img_size=img_size,
        in_channels=in_channels,
        seen_idx=base_class,
        all_idx=both_class,
        channels=in_channels,
        num_classes=num_classes,
        num_layers=3,
        num_heads=8,
        use_proj=False,
        use_stages=len(out_indices),
        embed_dims=in_channels,
        fusion_alpha=0.3,
        use_dynamic_fusion=True,
        use_atm_branch=True,
        cat_agg=dict(
            hidden_dim=128,
            num_layers=3,
            drop=0.1,
            use_temp_scale=True,
            use_cost_std=True,
            use_class_norm=True,
            use_class_linear=True,
            class_hidden_dim=128,
            class_nheads=2,
            class_layers=1,
            class_pool=(4, 4),
            class_attention='linear',
            class_pad_len=0,
            class_alpha=0.22,
            class_drop=0.2,
            text_guidance_dim=512,
            appearance_guidance_dim=512,
            appearance_guidance_proj_dim=128,
            decoder_dims=(64, 32),
            decoder_guidance_dims=(0, 0),
            decoder_guidance_proj_dims=(0, 0),
            prompt_channel=1,
            class_blocks_type='decoupled',
            class_num_prototypes=60,
        ),
        proto_div_weight=0,
        loss_decode=dict(
            type='SegLossPlus', num_classes=num_classes, dec_layers=3,
            mask_weight=100.0,
            dice_weight=1.0,
            loss_weight=1.0),
    ),
    test_cfg=dict(mode='slide', crop_size=(img_size, img_size), stride=(426, 426)), 
    base_class = base_class,
    novel_class = novel_class,
    both_class = both_class,
    ft_backbone = False,
    exclude_key='prompt',
    load_text_embedding='configs/_base_/datasets/text_embedding/context-59_multi.npy'
)

lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False,
                warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6)


optimizer = dict(type='AdamW', lr=0.00002, weight_decay=0.01,
        paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=10.0),
                                        'text_encoder': dict(lr_mult=0.0),
                                        'norm': dict(decay_mult=0.),
                                        'ln': dict(decay_mult=0.),
                                        'head': dict(lr_mult=10.),
                                        'decode_head.agg': dict(lr_mult=35.0),
                                        }))
data = dict(samples_per_gpu=12,
            workers_per_gpu=4,)

# Strict alignment with baseline effective batch and updates
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=10000)
evaluation = dict(interval=640001, metric='mIoU')

