from .vit import VisionTransformer
from .conemb_vit import Conemb_vit
from .vit_daff import Vit_daff
from .swin import SwinTransformer
from .cait import cait_models
from .conemb_vit_daff import VisionTransformer_conemb_daff
from .conemb_swin_daff import SwinTransformer_conemb_daff
from .conemb_cait_daff import cait_models_conemb_daff 
from .conemb_swin import Conemb_swin
from .swin_daff import Swin_daff
from .conemb_cait_impl import cait_models_conemb_impl
from functools import partial
from torch import nn


def create_model(img_size, n_classes, args):
    if args.arch == "vit":
        patch_size = 4 if img_size == 32 else 8  # 4 if img_size = 32 else 8
        model = VisionTransformer(img_size=[img_size],
                                  patch_size=args.patch_size,
                                  in_chans=3,
                                  num_classes=n_classes,
                                  embed_dim=192,
                                  depth=9,
                                  num_heads=12,
                                  mlp_ratio=args.vit_mlp_ratio,
                                  qkv_bias=True,
                                  drop_path_rate=args.sd,
                                  norm_layer=partial(nn.LayerNorm, eps=1e-6))

    elif args.arch == "conemb_vit_daff":       #conemb_vit_impl
        patch_size = 4 if img_size == 32 else 8  # 4 if img_size = 32 else 8
        model = VisionTransformer_conemb_daff(img_size=img_size,
                                              patch_size=args.patch_size,
                                              in_chans=3,
                                              num_classes=n_classes,
                                              embed_dim=192,
                                              depth=9,
                                              num_heads=12,
                                              mlp_ratio=args.vit_mlp_ratio,
                                              qkv_bias=True,
                                              drop_path_rate=args.sd,
                                              norm_layer=partial(nn.LayerNorm, eps=1e-6))
                                              
    elif args.arch == "conemb_vit":       #conemb_vit_impl
        patch_size = 4 if img_size == 32 else 8  # 4 if img_size = 32 else 8
        model = Conemb_vit(img_size=img_size,
                          patch_size=args.patch_size,
                          in_chans=3,
                          num_classes=n_classes,
                          embed_dim=192,
                          depth=9,
                          num_heads=12,
                          mlp_ratio=args.vit_mlp_ratio,
                          qkv_bias=True,
                          drop_path_rate=args.sd,
                          norm_layer=partial(nn.LayerNorm, eps=1e-6))
    elif args.arch == "vit_daff":       #conemb_vit_impl
        patch_size = 4 if img_size == 32 else 8  # 4 if img_size = 32 else 8
        model = Vit_daff(img_size=img_size,
                          patch_size=args.patch_size,
                          in_chans=3,
                          num_classes=n_classes,
                          embed_dim=192,
                          depth=9,
                          num_heads=12,
                          mlp_ratio=args.vit_mlp_ratio,
                          qkv_bias=True,
                          drop_path_rate=args.sd,
                          norm_layer=partial(nn.LayerNorm, eps=1e-6))
    
    elif args.arch == 'swin_daff': 
        mlp_ratio = args.vit_mlp_ratio
        window_size = 4
        patch_size = 2 if img_size == 32 else 4

        model = Swin_daff(img_size=img_size,
                                window_size=window_size, patch_size=patch_size, embed_dim=96, depths=[2, 6, 4],
                                num_heads=[3, 6, 12], num_classes=n_classes,
                                mlp_ratio=mlp_ratio, qkv_bias=True, drop_path_rate=args.sd)
    elif args.arch == 'conemb_swin': 
        mlp_ratio = args.vit_mlp_ratio
        window_size = 4
        patch_size = 2 if img_size == 32 else 4

        model = Conemb_swin(img_size=img_size,
                                window_size=window_size, patch_size=patch_size, embed_dim=96, depths=[2, 6, 4],
                                num_heads=[3, 6, 12], num_classes=n_classes,
                                mlp_ratio=mlp_ratio, qkv_bias=True, drop_path_rate=args.sd)  

    elif args.arch == 'cait':
        patch_size = 4 if img_size == 32 else 8
        model = cait_models(
            img_size=img_size, patch_size=patch_size, embed_dim=192, depth=24, num_heads=4,
            mlp_ratio=args.vit_mlp_ratio,
            qkv_bias=True, num_classes=n_classes, drop_path_rate=args.sd, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            init_scale=1e-5, depth_token_only=2)

    elif args.arch == 'conemb_cait_daff':      #conemb_cait_impl
        patch_size = 4 if img_size == 32 else 8
        model = cait_models_conemb_daff(
            img_size=img_size, patch_size=patch_size, embed_dim=192, depth=24, num_heads=4,
            mlp_ratio=args.vit_mlp_ratio,
            qkv_bias=True, num_classes=n_classes, drop_path_rate=args.sd, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            init_scale=1e-5, depth_token_only=2)

    elif args.arch == 'swin':

        mlp_ratio = args.vit_mlp_ratio
        window_size = 4
        patch_size = 2 if img_size == 32 else 4

        model = SwinTransformer(img_size=img_size,
                                window_size=window_size, patch_size=patch_size, embed_dim=96, depths=[2, 6, 4],
                                num_heads=[3, 6, 12], num_classes=n_classes,
                                mlp_ratio=mlp_ratio, qkv_bias=True, drop_path_rate=args.sd)

    elif args.arch == 'conemb_swin_daff':           #conemb_swin_impl

        mlp_ratio = args.vit_mlp_ratio
        window_size = 4
        patch_size = 2 if img_size == 32 else 4

        model = SwinTransformer_conemb_daff(img_size=img_size,
                                            window_size=window_size, patch_size=patch_size, embed_dim=96,
                                            depths=[2, 6, 4], num_heads=[3, 6, 12], num_classes=n_classes,
                                            mlp_ratio=mlp_ratio, qkv_bias=True, drop_path_rate=args.sd)

    elif args.arch == 'dhvt_vit':
        patch_size = 4 if img_size == 32 else 8  # 4 if img_size = 32 else 8
        model = VisionTransformer_dhvt(img_size=img_size,
                                       patch_size=args.patch_size,
                                       in_chans=3,
                                       num_classes=n_classes,
                                       embed_dim=192,
                                       depth=9,
                                       num_heads=12,
                                       mlp_ratio=args.vit_mlp_ratio,
                                       qkv_bias=True,
                                       drop_path_rate=args.sd,
                                       norm_layer=partial(nn.LayerNorm, eps=1e-6))

    elif args.arch == 'life_vit':
        patch_size = 4 if img_size == 32 else 8  # 4 if img_size = 32 else 8
        model = VisionTransformer_life(img_size=img_size,
                                       patch_size=args.patch_size,
                                       in_chans=3,
                                       num_classes=n_classes,
                                       embed_dim=192,
                                       depth=9,
                                       num_heads=12,
                                       mlp_ratio=args.vit_mlp_ratio,
                                       qkv_bias=True,
                                       drop_path_rate=args.sd,
                                       norm_layer=partial(nn.LayerNorm, eps=1e-6))

    else:
        NotImplementedError("Model architecture not implemented . . .")

    return model