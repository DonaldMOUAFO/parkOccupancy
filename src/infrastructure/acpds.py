import json
import torch
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from torch.utils.data import DataLoader
from functools import lru_cache

###### For tensorflow
import tensorflow as tf



class ACPDS():
    """
    A basic dataset of parking lot images,
    parking space coordinates (ROIs), and occupancies.
    Returns the tuple (image, rois, occupancy).
    """
    def __init__(self, dataset_path, ds_type='train', res=None):
        self.dataset_path = dataset_path
        self.ds_type = ds_type
        self.res = res

        # load all annotations
        with open(f'{self.dataset_path}/annotations.json', 'r') as f:
            all_annotations = json.load(f)

        # select train, valid, test, or all annotations
        if ds_type in ['train', 'valid', 'test']:
            # select train, valid, or test annotations
            annotations = all_annotations[ds_type]
        else:
            # select all annotations
            assert ds_type == 'all'
            # if using all annotations, combine the train, valid, and test dicts
            annotations = {k:[] for k in all_annotations['train'].keys()}
            for ds_type in ['train', 'valid', 'test']:
                for k, v in all_annotations[ds_type].items():
                    annotations[k] += v

        self.fname_list = annotations['file_names']
        self.rois_list = annotations['rois_list']
        self.occupancy_list = annotations['occupancy_list']

    @lru_cache(maxsize=None)
    def __getitem__(self, idx):
        # load image
        image_path = f'{self.dataset_path}/images/{self.fname_list[idx]}'
        image = torchvision.io.read_image(image_path)
        if self.res is not None:
            image = TF.resize(image, self.res)
            image = torch.tensor(image, dtype=torch.float16)
            
        # load occupancy
        occupancy = self.occupancy_list[idx]
        occupancy = torch.tensor(occupancy, dtype=torch.int64)
        #occupancy = torch.tensor(occupancy, dtype=torch.int32)

        # load rois
        rois = self.rois_list[idx]
        rois = torch.tensor(rois)
    
        return image, rois, occupancy
    
    def __len__(self):
        return len(self.fname_list)


def collate_fn(batch):
    images = [item[0] for item in batch]
    rois = [item[1] for item in batch]
    occupancy = [item[2] for item in batch]
    return [images, rois, occupancy]


def create_datasets(dataset_path, *args, **kwargs):
    """
    Create training and test DataLoaders.
    Returns the tuple (image, rois, occupancy).
    During the first pass, the DataLoaders will be cached.
    """
    ds_train = ACPDS(dataset_path, 'train', res=800, *args, **kwargs)
    ds_valid = ACPDS(dataset_path, 'valid', res=800, *args, **kwargs)
    ds_test  = ACPDS(dataset_path, 'test', res=800, *args, **kwargs)
    data_loader_train = DataLoader(ds_train, batch_size=1, shuffle=True, collate_fn=collate_fn)
    data_loader_valid = DataLoader(ds_valid, batch_size=1, shuffle=False, collate_fn=collate_fn)
    data_loader_test  = DataLoader(ds_test, batch_size=1, shuffle=False, collate_fn=collate_fn)
    return data_loader_train, data_loader_valid, data_loader_test

class ACPDS_TF():
    """
    A TensorFlow equivalent of the ACPDS dataset.
    Loads parking lot images, ROIs, and occupancy info.
    Returns (image, rois, occupancy) tuples.
    """
    def __init__(self, dataset_path, ds_type='train', res=None):
        self.dataset_path = dataset_path
        self.ds_type = ds_type
        self.res = res

        # Load all annotations
        with open(f'{self.dataset_path}/annotations.json', 'r') as f:
            all_annotations = json.load(f)
        
        # Select the subset
        if ds_type in ["train", "valid", "test"] :
            # select train, test or valid annotations
            annotations = all_annotations[ds_type]

        else :
            # select all annotations
            assert ds_type == 'all'
            # if using all annoations, combine the train, valid, and test dicts
            annotations = { k:[] for k in all_annotations["train"].keys()}
            for ds_type in ['train', 'valid', 'test']:
                for k, v in all_annotations[ds_type].items():
                    annotations[k] += v

        self.fname_list = annotations['file_names']
        self.rois_list = annotations['rois_list']
        self.occupancy_list = annotations['occupancy_list']

    def __len__(self):
        return len(self.fname_list)

    @lru_cache(maxsize=None)
    def _load_image(self, idx):
        """Load image and preprocess it."""
        image_path = f"{self.dataset_path}/images{self.fname_list[idx]}"
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3)
        # image = tf.image.convert_image_dtype(image, tf.float32)  # scale to [0, 1]

        if self.res is not None:
            image = tf.image.resize(image, self.res)
        
        # # load occupancy
        # occupancy = self.occupancy_list[idx]
        # #occupancy = tf.Tensor(occupancy, shape=(len(occupancy),), dtype=int32 )
        # occupancy = tf.convert_to_tensor(occupancy, dtype=tf.float32)

        # rois = self.rois_list[idx]
        # rois = tf.convert_to_tensor(rois)

        return image, # rois, occupancy

    def _get_item(self, idx):
        """Return one sample (image, rois, occupancy)."""

        image = self._load_image(idx)
        rois  = tf.convert_to_tensor( self.rois_list[idx], dtype=tf.float32 )
        occupancy = tf.convert_to_tensor(self.occupancy_list[idx], dtype=tf.int16)
        return image, rois, occupancy

    def as_tf_dataset(self, batch_size=1, shuffle=False):
        """Create a TensorFlow dataset."""
        ds = tf.data.Dataset.from_tensor_slices(tf.range(len(self.fname_list)))

        if shuffle:
            ds = ds.shuffle(buffer_size=len(self.fname_list))

        ds = ds.map(
            lambda idx: tf.py_function(
                func=lambda i: self._get_item(int(i)),
                inp=[idx],
                Tout=(tf.float32, tf.float32, tf.int16)
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

def create_datasets_tf(dataset_path, res=None):
    """Create training, validation, and test TensorFlow datasets."""
    ds_train = ACPDS_TF(dataset_path, 'train', res=res)
    ds_valid = ACPDS_TF(dataset_path, 'valid', res=res)
    ds_test = ACPDS_TF(dataset_path, 'test', res=res)

    tf_train = ds_train.as_tf_dataset(batch_size=1, shuffle=True)
    tf_valid = ds_valid.as_tf_dataset(batch_size=1, shuffle=False)
    tf_test = ds_test.as_tf_dataset(batch_size=1, shuffle=False)
    return tf_train, tf_valid, tf_test
