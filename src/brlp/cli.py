import os
import argparse
import yaml

import numpy as np
import torch
import pandas as pd
import nibabel as nib
from monai import transforms
from rich.console import Console
from rich.panel import Panel
from leaspy import Leaspy, AlgorithmSettings, Data
from brlp import networks, const, utils
from brlp import sample_using_controlnet_and_z


def infer():
    """
    """    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',              type=str, required=True, help='The path to the input CSV file.')
    parser.add_argument('--output',             type=str, required=True, help='The path to the folder where the output files will be saved.')
    parser.add_argument('--confs',              type=str, required=True, help='The path to the configuration file.')
    parser.add_argument('--target_age',         type=int, required=True, help='The last age of the predicted progression')
    parser.add_argument('--target_diagnosis',   type=int, required=True, help='The target subject\'s diagnosis (1=CN, 2=MCI, 3=AD)')
    parser.add_argument('--steps',              type=int, required=True, help='The number of timepoints between the starting and target age')
    parser.add_argument('--threads',            type=int, default=1,     help='The number of threads')
    parser.add_argument('--cpu',                action='store_true',     help='Enable this option to use the CPU for computation instead of the GPU.')
    args = parser.parse_args()

    console = Console()
    console.print(Panel("\n[white]If you use this work for your research, please cite:\n\nPuglisi, Lemuel, Daniel C. Alexander, and Daniele Ravì.\nEnhancing Spatiotemporal Disease Progression Models via Latent Diffusion and Prior Knowledge.\nMedical Image Computing and Computer Assisted Intervention, 2024.\n",
                        style="bold blue",
                        title="[bold orange]Brain Latent Progression", 
                        subtitle="[orange]Enhancing Spatiotemporal Disease Progression Models via Latent Diffusion and Prior Knowledge"))

    # selecting the torch device
    device = 'cpu' if args.cpu else 'cuda'

    # loading the configurations
    confs = yaml.safe_load(open(args.confs, 'r'))

    # select the auxiliary model type
    aux_type = { 1: 'cn', 2: 'mci', 3: 'ad' }
    aux_type = aux_type[args.target_diagnosis]

    #-------------------------
    #-------------------------

    with console.status("[bold green]Loading the models...") as status:
        # instantiate the networks.
        autoencoder = networks.init_autoencoder(confs['autoencoder']).to(device).eval()
        diffusion   = networks.init_latent_diffusion(confs['unet']).to(device).eval()
        controlnet  = networks.init_controlnet(confs['controlnet']).to(device).eval()
        auxiliary   = Leaspy.load(confs['aux'][aux_type])
    console.print("✅ [bold green]Models loaded.")

    # load the input
    input_df = pd.read_csv(args.input).sort_values('age')
    segm_dir = os.path.join(args.output, 'segmentations')
    os.makedirs(segm_dir, exist_ok=True)

    #-------------------------
    #-------------------------

    with console.status("[bold green]Extracting AD-related regional volumes from past brain MRIs") as status:
        records = []
        for input_row in input_df.itertuples():

            if 'segm_path' not in input_df.columns:
                scan_path = input_row.image_path
                segm_path = os.path.join(segm_dir, f'{input_row.image_uid}_segm.nii.gz')
                os.system(f'mri_synthseg --i {scan_path} --o {segm_path} --fast --threads {args.threads}')
            else:
                segm_path = input_row.segm_path

            measurements = _measure_synthseg(segm_path, confs)
            records.append({ 'ID': 'pt', 'TIME': input_row.age } | measurements)

        patient_records = pd.DataFrame(records)
        patient_records = _map_to_data(patient_records)
    console.print("✅ [bold green]Regional volumes successfully extracted from available brain MRIs")
    
    #-------------------------
    #-------------------------

    with console.status("[bold green]Using prior knowledge about disease progression to estimate volumetric trajectories") as status:
        auxiliary_settings = AlgorithmSettings('scipy_minimize')
        ip = auxiliary.personalize(patient_records, auxiliary_settings)
        last_record = input_df.iloc[-1]
        last_age = last_record.age
        timepoints = np.linspace(last_age, args.target_age, args.steps)
        estimates = auxiliary.estimate({ 'pt': timepoints }, ip)['pt']
        estimates = _reverse_and_correct(estimates, confs)
    console.print("✅ [bold green]Volumetric trajectories estimated successfully")

    load_volume_tensor = transforms.Compose([
        transforms.CopyItemsD(keys={'image_path'}, names=['image']),
        transforms.LoadImageD(image_only=True, keys=['image']),
        transforms.EnsureChannelFirstD(keys=['image']), 
        transforms.SpacingD(pixdim=const.RESOLUTION, keys=['image']),
        transforms.ResizeWithPadOrCropD(spatial_size=const.INPUT_SHAPE_AE, mode='minimum', keys=['image']),
        transforms.ScaleIntensityD(minv=0, maxv=1, keys=['image'])
    ])

    input_image = load_volume_tensor({ 'image_path': last_record.image_path })['image']
    input_image = input_image.unsqueeze(0).to(device)
    input_latent = autoencoder.encode(input_image)[0]
    input_latent = transforms.DivisiblePad(k=4, mode='constant')(input_latent.squeeze(0))


    with console.status("[bold green]Predicting future brain MRIs") as status:

        for i, target_age in enumerate(timepoints):

            # subject-specific covariates
            s = [(target_age - const.AGE_MIN) / const.AGE_DELTA,
                (last_record.sex - const.SEX_MIN) / const.SEX_DELTA,
                (args.target_diagnosis - const.DIA_MIN) / const.DIA_DELTA]

            # progression-related covariates
            v = list(estimates[i])

            # full set of covariates
            covariates = torch.tensor(s + v)

            # predict the followup MRI
            mri = sample_using_controlnet_and_z(
                autoencoder=autoencoder, 
                diffusion=diffusion, 
                controlnet=controlnet, 
                starting_z=input_latent.float(), 
                starting_a=last_record.age / 100, 
                context=covariates.float(), 
                device=device, 
                scale_factor=1,
                average_over_n=confs['las']['m'],
                num_inference_steps=25, 
                verbose=False
            )

            # save the image
            mri = nib.Nifti1Image(mri.numpy(), affine=const.MNI152_1P5MM_AFFINE)
            mri = utils.percnorm_nifti(mri, 5, 99.5)
            mri = utils.apply_mask(mri, nib.load(segm_path))
            mri.to_filename(os.path.join(args.output, f'followup_age_{target_age:.0f}.nii.gz'))
            console.log(f"Brain MRI at age {int(target_age)} successfully generated.")

    console.print("[bold green] All the requested brain MRIs have been generated.")


def _measure_synthseg(segm_path, confs):
    """
    Measure synthseg segmentation and return just
    coarse regions of interest, normalized using the 
    training set statistics.
    """
    segm = nib.load(segm_path).get_fdata().round()

    full_regions = set()
    regions_names = [ r for r in const.SYNTHSEG_CODEMAP.values() if 'background' not in r ]

    for region in regions_names:
        full_regions.add( region.replace('left_', '').replace('right_', '') )
    
    record = {}
    for region in full_regions: 
        if region in const.CONDITIONING_VARIABLES:
            record[region] = 0
    
    for code, region in const.SYNTHSEG_CODEMAP.items():
        fregion = region.replace('left_', '').replace('right_', '')
        if fregion in const.CONDITIONING_VARIABLES:
            record[fregion] += (segm == code).sum()
    
    for region, raw_v in record.items():
        _min, _max = confs['minmax_params'][region]
        record[region] = (raw_v - _min) / (_max - _min)
    return record


def _map_to_data(df):
    """
    Convert Pandas dataframe to Leaspy data object.
    """
    df = df.copy()

    if df.iloc[0].TIME >= 0 and df.iloc[0].TIME <= 1:
        df['TIME'] *= 100

    if 'months_to_screening' in df.columns:
        df['age'] += df['months_to_screening'] / 1000 # dirty trick to avoid duplicate visits because of same age
    else:
        df = df.drop_duplicates(['ID', 'TIME'], keep='first')

    for region in const.CONDITIONING_REGIONS: 
        if region == 'lateral_ventricle': continue # invert if decreasing    
        df[region] = 1 - df[region]

    df = df.set_index(["ID","TIME"], verify_integrity=False).sort_index()
    df = df[const.CONDITIONING_REGIONS]
    return Data.from_dataframe(df)


def _reverse_and_correct(estimates, confs):
    """
    """
    for i in range(len(estimates)):
        for j, region_name in enumerate(const.CONDITIONING_REGIONS):
            if region_name != 'lateral_ventricle':
                estimates[i][j] = 1 - estimates[i][j]
            x = estimates[i][j]
            m, q = confs['median_corrections'][region_name]
            x = (x - q) / m
            x = max(x, -0.2)
            x = min(x,  1.2)
            estimates[i][j] = x
    return estimates