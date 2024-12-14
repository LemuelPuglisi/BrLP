# Data preparation and study reproducibility

## Table of Contents
- [Downloading the data](#downloading-the-data)
- [Preprocessing the MRIs](#preprocessing-the-mris)
- [Organizing the dataset](#organizing-the-dataset)
- [Reproducibility inquiries](#reproducibility-inquiries)

## Downloading the data

Please proceed to [the next section](#preprocessing-the-mris) if you have your own longitudinal data.


All of the datasets used in our study are publicly accessible; however, accessing them requires an application process. Here we provide some useful links. 

| Dataset | Dataset website                            | Application link                                            |
| ------- | ------------------------------------------ | ----------------------------------------------------------- |
| ADNI    | [link](https://adni.loni.usc.edu/)         | [link](https://adni.loni.usc.edu/data-samples/access-data/) |
| OASIS 3 | [link](https://www.oasis-brains.org/#data) | [link](https://www.oasis-brains.org/#data)                  |
| AIBL    | [link](https://aibl.org.au/)               | [link](https://ida.loni.usc.edu/login.jsp)                  |

Upon gaining access, download all T1-weighted MRIs, retaining only the data for subjects with multiple visits. Both demographic data (age, sex) and diagnostic information (cognitive status) are accessible across all three datasets. The following decision trees were utilized to extract diagnostic information from the ADNI and OASIS 3 datasets. For AIBL, the ADNI tree can be used.

![decision-tree](assets/diagnosis_decision_tree.drawio.svg)


## Preprocessing the MRIs

Each MRI undergoes preprocessing. Assuming that your MRIs are already in the nifti format, perform the following steps:

| Step                                 | Algorithm    | Command                       | Software                                                     |
| ------------------------------------ | ------------ | ----------------------------- | ------------------------------------------------------------ |
| bias-field correction                | N4           | `N4BiasFieldCorrection`       | [ANTs](https://github.com/ANTsX/ANTs)                        |
| skull stripping                      | SynthStrip   | `mri_synthstrip`              | [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/)            |
| Affine registration to MNI 152 space | SyN          | `antsRegistrationSyNQuick.sh` | [ANTs](https://github.com/ANTsX/ANTs)                        |
| Brain regions segmentation           | SynthSeg 2.0 | `mri_synthseg`                | [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/)            |
| Intensity normalization              | WhiteStripe  | `ws-normalize`                | [intensity-normalization](https://github.com/jcreinhold/intensity-normalization) |

or just run the `turboprep` pipeline ([see here](https://github.com/LemuelPuglisi/turboprep)), which executes the same steps:

```bash
turboprep <image_path> <output_folder> </path/to/MNI152template> -m t1
```

You can get the MNI152 template file [here](https://github.com/Washington-University/HCPpipelines/blob/master/global/templates/MNI152_T1_1mm_brain.nii.gz). The preprocessing outputs a normalized, skull-stripped, and corrected brain MRI along with its segmentation. All these steps take approximately 30 seconds to complete (tested on a GeForce RTX 4090 GPU).



## Organizing the dataset

Organize your data into a CSV file (`dataset.csv`) with 9 fields in total:

* `subject_id`: the unique identifier of the subject
* `image_uid`: the unique identifier of the brain MRI
* `split`: `train`, `valid` or `test`
* `sex`: the sex of the subject (male=0, female=1)
* `age`: age of the subject (divided by 100) when the MRI was acquired 
* `diagnosis`: cognitive status (CN=0, MCI=0.5, AD=1) 
* `last_diagnosis`: last `diagnosis` available among all follow-up visits 
* `image_path`: absolute path to the brain MRI
* `segm_path`: absolute path to the brain MRI segmentation
* `latent_path`: absolute path to the brain MRI latent (*)

(*) The latents will be extracted following the training of the autoencoder. We provide a script that automatically extracts latents from your entire dataset. This script stores the latent representations in the same folder as the images, replacing the '.nii.gz' extension with '_latent.npz' in the filename. As a result, you can anticipate the storage location of the latent representations even before their computation. For instance, if `image_path=/foo/bar/brainXYZ.nii.gz`, the corresponding latent will be stored at `/foo/bar/brainXYZ_latent.npz`.

### Computing CSVs for training

The autoencoder can be trained directly using the dataset.csv file. The Diffusion UNet requires knowledge of the regional volumes associated with each brain MRI, which will be provided in a CSV file named `A.csv`. The ControlNet necessitates pairs of images, $x_A$ and $x_B$, acquired at ages $A < B$ from the same subject. These records will be organized in pairs within a dataset named B.csv. To generate `A.csv` and `B.csv`, execute the following command:

```bash
python scripts/prepare/prepare_csv.py \
    --dataset_csv dataset.csv \
    --output_path /path/to/output/dir
```

## Reproducibility inquiries

For further inquiries regarding reproducibility, please open a GitHub issue.
