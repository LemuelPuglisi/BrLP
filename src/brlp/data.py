import os
from typing import Optional, Union

import pandas as pd
from monai.data import Dataset, PersistentDataset
from monai.transforms.transform import Transform


def get_dataset_from_pd(df: pd.DataFrame, transforms_fn: Transform, cache_dir: Optional[str]) -> Union[Dataset,PersistentDataset]: 
    """
    If `cache_dir` is defined, returns a `monai.data.PersistenDataset`. 
    Otherwise, returns a simple `monai.data.Dataset`.

    Args:
        df (pd.DataFrame): Dataframe describing each image in the longitudinal dataset.
        transforms_fn (Transform): Set of transformations
        cache_dir (Optional[str]): Cache directory (ensure enough storage is available)

    Returns:
        Dataset|PersistentDataset: The dataset
    """
    assert cache_dir is None or os.path.exists(cache_dir), 'Invalid cache directory path'
    data = df.to_dict(orient='records')
    return Dataset(data=data, transform=transforms_fn) if cache_dir is None \
        else PersistentDataset(data=data, transform=transforms_fn, cache_dir=cache_dir)