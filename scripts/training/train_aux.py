import os
import argparse
import warnings

import pandas as pd
from leaspy import Leaspy, Data, AlgorithmSettings
from brlp import const



def prepare_dcm_data(df):
    """
    Convert dataframe to leaspy.Data according to Leaspy formatting rules, which are:
    i)   name `subject_id` as `ID` and `age` as `TIME` and use this pair as row unique index.
    ii)  all the biomarkers should be normalized between [0,1].
    iii) all the biomarkers should be increasing with time, invert the trend if not.  
    """
    df = df.copy()
    
    # age should be unnormalized.
    df['age'] = df['age'] * 100 

    # if the patient has 2 scans at the same age, we will
    # have more than one row with the same index in our dataframe.
    # Use something as the months to the screening visits to distinguish
    # different visits at the same age. 
    if 'months_to_screening' in df.columns:
        df['age'] += df['months_to_screening'] / 1000

    # only the lateral ventricle volume increases over time among
    # the five selected regions. Invert the trend where needed.
    for region in const.CONDITIONING_REGIONS:
        if region == 'lateral_ventricle': continue    
        df[region] = 1 - df[region]

    # set the age and subject id as indices.
    df = df.rename({'subject_id': 'ID', 'age': 'TIME'}, axis=1) \
           .set_index(["ID","TIME"], verify_integrity=False) \
           .sort_index()

    # select only the regions to model.
    df = df[const.CONDITIONING_REGIONS]
    return Data.from_dataframe(df)
    

def train_leaspy(data, name, logs_path):
    """
    Train a DCM logistic model with provided data.
    """
    model_logs_path = os.path.join(logs_path, f'leaspy_logs_{name}')
    os.makedirs(model_logs_path, exist_ok=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        leaspy = Leaspy("logistic", source_dimension=4)
        algo_settings = AlgorithmSettings('mcmc_saem', n_iter=5000, seed=64)
        algo_settings.set_logs(
            path=model_logs_path,          
            plot_periodicity=50,  
            save_periodicity=10,  
            console_print_periodicity=None,  
            overwrite_logs_folder=True
        )
        leaspy.fit(data, settings=algo_settings)
        return leaspy


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_csv', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.dataset_csv)
    train_df = df[ df.split == 'train' ]

    logs_path = os.path.join(args.output_path, 'logs')
    os.makedirs(logs_path, exist_ok=True)

    # train DCM on cognitively normal subjects.
    print('>\033[1m Training DCM on cognitively normal subjects \033[0m')
    cn_df       = train_df[train_df.last_diagnosis == 0.]
    cn_data     = prepare_dcm_data(cn_df)
    cn_leaspy   = train_leaspy(cn_data, 'cn', logs_path)
    cn_leaspy.save(os.path.join(args.output_path, 'dcm_cn.json'))

    # train DCM on subjects with mild cognitive impairment.
    print('>\033[1m Training DCM on subjects with Mild Cognitive Impairment \033[0m')
    mci_df       = train_df[train_df.last_diagnosis == 0.5]
    mci_data     = prepare_dcm_data(mci_df)
    mci_leaspy   = train_leaspy(mci_data, 'mci', logs_path)
    mci_leaspy.save(os.path.join(args.output_path, 'dcm_mci.json'))

    # train DCM on subjects with Alzheimer's disease.
    print('>\033[1m Training DCM on subjects with Alzheimer\'s disease \033[0m')
    ad_df       = train_df[train_df.last_diagnosis == 1.]
    ad_data     = prepare_dcm_data(ad_df)
    ad_leaspy   = train_leaspy(ad_data, 'ad', logs_path)
    ad_leaspy.save(os.path.join(args.output_path, 'dcm_ad.json'))