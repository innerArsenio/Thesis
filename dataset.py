import os
import json

import torch
from torch.utils.data import Dataset
import numpy as np

from PIL import Image
import PIL.Image
try:
    import pyspng
except ImportError:
    pyspng = None
from sklearn.model_selection import train_test_split

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None, mode="train"):
        PIL.Image.init()
        supported_ext = PIL.Image.EXTENSION.keys() | {'.npy'}
        self.mode = mode
        self.transforms = transform

        if self.mode=="test":
            self.images_dir = "/home/arsen.abzhanov/Thesis/REPA/ISIC_2018_REPA/test/images"
            self.features_dir = "/home/arsen.abzhanov/Thesis/REPA/ISIC_2018_REPA/test/vae-sd"
        else:
            self.images_dir = os.path.join(data_dir, 'images')
            self.features_dir = os.path.join(data_dir, 'vae-sd')

        # images
        self._image_fnames = {
            os.path.relpath(os.path.join(root, fname), start=self.images_dir)
            for root, _dirs, files in os.walk(self.images_dir) for fname in files
            }
        self.image_fnames = sorted(
            fname for fname in self._image_fnames if self._file_ext(fname) in supported_ext
            )
        # features
        self._feature_fnames = {
            os.path.relpath(os.path.join(root, fname), start=self.features_dir)
            for root, _dirs, files in os.walk(self.features_dir) for fname in files
            }
        self.feature_fnames = sorted(
            fname for fname in self._feature_fnames if self._file_ext(fname) in supported_ext
            )
        # labels
        fname = 'dataset.json'
        with open(os.path.join(self.features_dir, fname), 'rb') as f:
            labels = json.load(f)['labels']
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self.feature_fnames]
        labels = np.array(labels)
        self.labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])

        # Get unique classes
        unique_classes = np.unique(self.labels)

        if self.mode=="test":
            print("test mode")
        else:
            # Initialize lists to hold training and validation data
            train_indices = []
            val_indices = []

            # Split each class
            for cls in unique_classes:
                # Get indices of all samples in the current class
                cls_indices = np.where(self.labels == cls)[0]
                
                # Split the indices into training and validation sets
                cls_train_indices, cls_val_indices = train_test_split(cls_indices, test_size=0.1, random_state=42)
                
                # Append to the respective lists
                train_indices.extend(cls_train_indices)
                val_indices.extend(cls_val_indices)

            # Convert lists to numpy arrays
            train_indices = np.array(train_indices)
            val_indices = np.array(val_indices)

            if self.mode=="train":
                self.image_fnames = [self.image_fnames[i] for i in train_indices]
                self.feature_fnames = [self.feature_fnames[i] for i in train_indices]
                self.labels = self.labels[train_indices]
            elif self.mode == "val":
                self.image_fnames = [self.image_fnames[i] for i in val_indices]
                self.feature_fnames = [self.feature_fnames[i] for i in val_indices]
                self.labels = self.labels[val_indices]

        # Print the number of samples in each set
        print(f"Number of samples: {len(self.image_fnames)}")

    def _file_ext(self, fname):
        return os.path.splitext(fname)[1].lower()

    def __len__(self):
        assert len(self.image_fnames) == len(self.feature_fnames), \
            "Number of feature files and label files should be same"
        return len(self.feature_fnames)

    def __getitem__(self, idx):
        image_fname = self.image_fnames[idx]
        feature_fname = self.feature_fnames[idx]
        image_ext = self._file_ext(image_fname)
        with open(os.path.join(self.images_dir, image_fname), 'rb') as f:
            if image_ext == '.npy':
                image = np.load(f)
                image = image.reshape(-1, *image.shape[-2:])
            elif image_ext == '.png' and pyspng is not None:
                image = pyspng.load(f.read())
                image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)
            else:
                image = np.array(PIL.Image.open(f))
                image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)

        features = np.load(os.path.join(self.features_dir, feature_fname))
        if self.transforms is not None:
            #image, features = self.transforms(torch.from_numpy(image), torch.from_numpy(features))
            return self.transforms((torch.from_numpy(image), torch.from_numpy(features))), torch.tensor(self.labels[idx])
        else:
            return torch.from_numpy(image), torch.from_numpy(features), torch.tensor(self.labels[idx])