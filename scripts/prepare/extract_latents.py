import os
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from monai import transforms
from brlp import init_autoencoder
from brlp import const


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_csv', type=str, required=True)
    parser.add_argument('--aekl_ckpt',   type=str, required=True)
    args = parser.parse_args()

    autoencoder = init_autoencoder(args.aekl_ckpt).to(DEVICE).eval()

    transforms_fn = transforms.Compose([
        transforms.CopyItemsD(keys={'image_path'}, names=['image']),
        transforms.LoadImageD(image_only=True, keys=['image']),
        transforms.EnsureChannelFirstD(keys=['image']), 
        transforms.SpacingD(pixdim=const.RESOLUTION, keys=['image']),
        transforms.ResizeWithPadOrCropD(spatial_size=const.INPUT_SHAPE_AE, mode='minimum', keys=['image']),
        transforms.ScaleIntensityD(minv=0, maxv=1, keys=['image'])
    ])
    
    df = pd.read_csv(args.dataset_csv)

    with torch.no_grad():
        for image_path in tqdm(df.image_path, total=len(df)):
            destpath = image_path.replace('.nii.gz', '_latent.npz').replace('.nii', '_latent.npz')            
            if os.path.exists(destpath): continue
            mri_tensor = transforms_fn({'image_path': image_path})['image'].to(DEVICE)
            mri_latent, _ = autoencoder.encode(mri_tensor.unsqueeze(0))
            mri_latent = mri_latent.cpu().squeeze(0).numpy()
            np.savez_compressed(destpath, data=mri_latent)