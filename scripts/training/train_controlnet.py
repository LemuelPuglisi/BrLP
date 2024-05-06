import os
import argparse
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import nibabel as nib
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from monai import transforms
from monai.data.image_reader import NumpyReader
from generative.networks.schedulers import DDPMScheduler
from tqdm import tqdm

from brlp import const
from brlp import utils
from brlp import networks
from brlp import (
    get_dataset_from_pd, 
    sample_using_controlnet_and_z
)


warnings.filterwarnings("ignore")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def concat_covariates(_dict):
    """
    Provide context for cross-attention layers and concatenate the
    covariates in the channel dimension.
    """
    conditions = [
        _dict['followup_age'], 
        _dict['sex'], 
        _dict['followup_diagnosis'], 
        _dict['followup_cerebral_cortex'], 
        _dict['followup_hippocampus'], 
        _dict['followup_amygdala'], 
        _dict['followup_cerebral_white_matter'], 
        _dict['followup_lateral_ventricle']        
    ]
    _dict['context'] = torch.tensor(conditions).unsqueeze(0)
    return _dict


def images_to_tensorboard(
    writer,
    epoch, 
    mode, 
    autoencoder, 
    diffusion, 
    controlnet, 
    dataset,
    scale_factor
):
    """
    Visualize the generation on tensorboard
    """
    resample_fn = transforms.Spacing(pixdim=1.5)
    random_indices = np.random.choice( range(len(dataset)), 3 ) 

    for tag_i, i in enumerate(random_indices):

        starting_z = dataset[i]['starting_latent'] * scale_factor
        context    = dataset[i]['context']
        starting_a = dataset[i]['starting_age']

        starting_image = torch.from_numpy(nib.load(dataset[i]['starting_image']).get_fdata()).unsqueeze(0)
        followup_image = torch.from_numpy(nib.load(dataset[i]['followup_image']).get_fdata()).unsqueeze(0)
        starting_image = resample_fn(starting_image).squeeze(0)
        followup_image = resample_fn(followup_image).squeeze(0)

        predicted_image = sample_using_controlnet_and_z(
            autoencoder=autoencoder, 
            diffusion=diffusion, 
            controlnet=controlnet, 
            starting_z=starting_z, 
            starting_a=starting_a, 
            context=context, 
            device=DEVICE,
            scale_factor=scale_factor
        )

        utils.tb_display_cond_generation(
            writer=writer, 
            step=epoch, 
            tag=f'{mode}/comparison_{tag_i}',
            starting_image=starting_image, 
            followup_image=followup_image, 
            predicted_image=predicted_image
        )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_csv', required=True,   type=str)
    parser.add_argument('--cache_dir',   required=True,   type=str)
    parser.add_argument('--output_dir',  required=True,   type=str)
    parser.add_argument('--aekl_ckpt',   required=True,   type=str)
    parser.add_argument('--diff_ckpt',   required=True,   type=str)
    parser.add_argument('--cnet_ckpt',   default=None,    type=str)
    parser.add_argument('--num_workers', default=8,       type=int)
    parser.add_argument('--n_epochs',    default=5,       type=int)
    parser.add_argument('--batch_size',  default=16,      type=int)
    parser.add_argument('--lr',          default=2.5e-5,  type=float)
    
    args = parser.parse_args()


    npz_reader = NumpyReader(npz_keys=['data'])
    transforms_fn = transforms.Compose([
        transforms.LoadImageD(keys=['starting_latent', 'followup_latent'], reader=npz_reader), 
        transforms.EnsureChannelFirstD(keys=['starting_latent', 'followup_latent'], channel_dim=0), 
        transforms.DivisiblePadD(keys=['starting_latent', 'followup_latent'], k=4, mode='constant'), 
        transforms.Lambda(func=concat_covariates),
    ])

    dataset_df = pd.read_csv(args.dataset_csv)
    train_df = dataset_df[ dataset_df.split == 'train' ]
    valid_df = dataset_df[ dataset_df.split == 'valid' ]
    trainset = get_dataset_from_pd(train_df, transforms_fn, args.cache_dir)
    validset = get_dataset_from_pd(valid_df, transforms_fn, args.cache_dir)

    train_loader = DataLoader(dataset=trainset, 
                              num_workers=args.num_workers, 
                              batch_size=args.batch_size, 
                              shuffle=True, 
                              persistent_workers=True, 
                              pin_memory=True)

    valid_loader = DataLoader(dataset=validset, 
                              num_workers=args.num_workers, 
                              batch_size=args.batch_size, 
                              shuffle=True, 
                              persistent_workers=True, 
                              pin_memory=True)

    autoencoder = networks.init_autoencoder(args.aekl_ckpt)
    diffusion   = networks.init_latent_diffusion(args.diff_ckpt)
    controlnet  = networks.init_controlnet()

    if args.cnet_ckpt is not None:
        print('Resuming training...')
        controlnet.load_state_dict(torch.load(args.cnet_ckpt))
    else:
        print('Copying weights from diffusion model')
        controlnet.load_state_dict(diffusion.state_dict(), strict=False)

    # freeze the unet weights
    for p in diffusion.parameters():
        p.requires_grad = False

    # Move everything to DEVICE
    autoencoder.to(DEVICE)
    diffusion.to(DEVICE)
    controlnet.to(DEVICE)

    scaler = GradScaler()
    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=args.lr)

    with torch.no_grad():
        with autocast(enabled=True):
            z = trainset[0]['followup_latent']

    scale_factor = 1 / torch.std(z)
    print(f"Scaling factor set to {scale_factor}")

    scheduler = DDPMScheduler(num_train_timesteps=1000, 
                              schedule='scaled_linear_beta', 
                              beta_start=0.0015, 
                              beta_end=0.0205)
    
    writer = SummaryWriter()

    global_counter  = { 'train': 0, 'valid': 0 }
    loaders         = { 'train': train_loader, 'valid': valid_loader }
    datasets        = { 'train': trainset, 'valid': validset }


    for epoch in range(args.n_epochs):
        
        for mode in loaders.keys():
            print('mode:', mode)
            loader = loaders[mode]
            controlnet.train() if mode == 'train' else controlnet.eval()
            epoch_loss = 0.
            progress_bar = tqdm(enumerate(loader), total=len(loader))
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in progress_bar:
                
                if mode == 'train':
                    optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(mode == 'train'):

                    starting_z = batch['starting_latent'].to(DEVICE)  * scale_factor
                    followup_z = batch['followup_latent'].to(DEVICE)  * scale_factor
                    context    = batch['context'].to(DEVICE)
                    starting_a = batch['starting_age'].to(DEVICE)

                    n = starting_z.shape[0] 

                    with autocast(enabled=True):

                        concatenating_age      = starting_a.view(n, 1, 1, 1, 1).expand(n, 1, *starting_z.shape[-3:])
                        controlnet_condition   = torch.cat([ starting_z, concatenating_age ], dim=1)

                        noise = torch.randn_like(followup_z).to(DEVICE)
                        timesteps = torch.randint(0, scheduler.num_train_timesteps, (n,), device=DEVICE).long()
                        images_noised = scheduler.add_noise(followup_z, noise=noise, timesteps=timesteps)

                        down_h, mid_h = controlnet(
                            x=images_noised.float(), 
                            timesteps=timesteps, 
                            context=context.float(),
                            controlnet_cond=controlnet_condition.float()
                        )

                        noise_pred = diffusion(
                            x=images_noised.float(), 
                            timesteps=timesteps, 
                            context=context.float(), 
                            down_block_additional_residuals=down_h,
                            mid_block_additional_residual=mid_h
                        )

                        loss = F.mse_loss(noise_pred.float(), noise.float())

                if mode == 'train':
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                
                #-------------------------------
                # Iteration end
                writer.add_scalar(f'{mode}/batch-mse', loss.item(), global_counter[mode])
                epoch_loss += loss.item()
                progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
                global_counter[mode] += 1

            # Epoch loss
            epoch_loss = epoch_loss / len(loader)
            writer.add_scalar(f'{mode}/epoch-mse', epoch_loss, epoch)
            
            # Logging visualization
            images_to_tensorboard(
                writer=writer,
                epoch=epoch,
                mode=mode, 
                autoencoder=autoencoder, 
                diffusion=diffusion, 
                controlnet=controlnet,
                dataset=datasets[mode], 
                scale_factor=scale_factor
            )

        if epoch > 2:
            savepath = os.path.join(args.output_dir, f'cnet-ep-{epoch}.pth')
            torch.save(controlnet.state_dict(), savepath)