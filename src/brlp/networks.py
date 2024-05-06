import os
from typing import Optional

import torch
import torch.nn as nn
from generative.networks.nets import (
    AutoencoderKL, 
    PatchDiscriminator,
    DiffusionModelUNet, 
    ControlNet
)


def load_if(checkpoints_path: Optional[str], network: nn.Module) -> nn.Module:
    """
    Load pretrained weights if available.

    Args:
        checkpoints_path (Optional[str]): path of the checkpoints
        network (nn.Module): the neural network to initialize 

    Returns:
        nn.Module: the initialized neural network
    """
    if checkpoints_path is not None:
        assert os.path.exists(checkpoints_path), 'Invalid path'
        network.load_state_dict(torch.load(checkpoints_path))
    return network


def init_autoencoder(checkpoints_path: Optional[str] = None) -> nn.Module:
    """
    Load the KL autoencoder (pretrained if `checkpoints_path` points to previous params).

    Args:
        checkpoints_path (Optional[str], optional): path of the checkpoints. Defaults to None.

    Returns:
        nn.Module: the KL autoencoder
    """
    autoencoder = AutoencoderKL(spatial_dims=3, 
                                in_channels=1, 
                                out_channels=1, 
                                latent_channels=3,
                                num_channels=(64, 128, 128, 128),
                                num_res_blocks=2, 
                                norm_num_groups=32,
                                norm_eps=1e-06,
                                attention_levels=(False, False, False, False), 
                                with_decoder_nonlocal_attn=False, 
                                with_encoder_nonlocal_attn=False)
    return load_if(checkpoints_path, autoencoder)


def init_patch_discriminator(checkpoints_path: Optional[str] = None) -> nn.Module:
    """
    Load the patch discriminator (pretrained if `checkpoints_path` points to previous params).

    Args:
        checkpoints_path (Optional[str], optional): path of the checkpoints. Defaults to None.

    Returns:
        nn.Module: the parch discriminator
    """
    patch_discriminator = PatchDiscriminator(spatial_dims=3, 
                                             num_layers_d=3, 
                                             num_channels=32, 
                                             in_channels=1, 
                                             out_channels=1)
    return load_if(checkpoints_path, patch_discriminator)

    
def init_latent_diffusion(checkpoints_path: Optional[str] = None) -> nn.Module:
    """
    Load the UNet from the diffusion model (pretrained if `checkpoints_path` points to previous params).

    Args:
        checkpoints_path (Optional[str], optional): path of the checkpoints. Defaults to None.

    Returns:
        nn.Module: the UNet
    """
    latent_diffusion = DiffusionModelUNet(spatial_dims=3, 
                                          in_channels=3, 
                                          out_channels=3, 
                                          num_res_blocks=2, 
                                          num_channels=(256, 512, 768), 
                                          attention_levels=(False, True, True), 
                                          norm_num_groups=32, 
                                          norm_eps=1e-6, 
                                          resblock_updown=True, 
                                          num_head_channels=(0, 512, 768), 
                                          transformer_num_layers=1,
                                          with_conditioning=True,
                                          cross_attention_dim=8,
                                          num_class_embeds=None, 
                                          upcast_attention=True, 
                                          use_flash_attention=False)
    return load_if(checkpoints_path, latent_diffusion)


def init_controlnet(checkpoints_path: Optional[str] = None) -> nn.Module:
    """
    Load the ControlNet (pretrained if `checkpoints_path` points to previous params).

    Args:
        checkpoints_path (Optional[str], optional): path of the checkpoints. Defaults to None.

    Returns:
        nn.Module: the ControlNet
    """
    controlnet = ControlNet(spatial_dims=3, 
                            in_channels=3,
                            num_res_blocks=2, 
                            num_channels=(256, 512, 768), 
                            attention_levels=(False, True, True), 
                            norm_num_groups=32, 
                            norm_eps=1e-6, 
                            resblock_updown=True, 
                            num_head_channels=(0, 512, 768), 
                            transformer_num_layers=1, 
                            with_conditioning=True,
                            cross_attention_dim=8, 
                            num_class_embeds=None, 
                            upcast_attention=True, 
                            use_flash_attention=False, 
                            conditioning_embedding_in_channels=4,  
                            conditioning_embedding_num_channels=(256,))
    return load_if(checkpoints_path, controlnet)
