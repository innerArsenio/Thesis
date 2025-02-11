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
import random

LIST_OF_TASKS = ['ISIC', 'IDRID', 'BUSI', 'BUSI_SOFT']
TASK='ISIC'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(1)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None, mode="train"):
        PIL.Image.init()
        supported_ext = PIL.Image.EXTENSION.keys() | {'.npy'}
        self.mode = mode
        self.transforms = transform

        self.seed = 42
        #set_seed(self.seed)

        # if self.mode=="test":
        #     self.images_dir = "/home/arsen.abzhanov/Thesis_local/REPA/ISIC_2018_REPA/test/images"
        #     self.features_dir = "/home/arsen.abzhanov/Thesis_local/REPA/ISIC_2018_REPA/test/vae-sd"
        # elif self.mode=="val":
        #     self.images_dir = "/home/arsen.abzhanov/Thesis_local/REPA/ISIC_2018_REPA/validation/images"
        #     self.features_dir = "/home/arsen.abzhanov/Thesis_local/REPA/ISIC_2018_REPA/validation/vae-sd"
        # elif self.mode=="train":
        #     self.images_dir = os.path.join(data_dir, 'images')
        #     self.features_dir = os.path.join(data_dir, 'vae-sd')
        # elif self.mode=="muddled_0_025":
        #     self.images_dir = "/home/arsen.abzhanov/Thesis_local/REPA/ISIC_2018_muddled_0_025_REPA/images"
        #     self.features_dir = "/home/arsen.abzhanov/Thesis_local/REPA/ISIC_2018_muddled_0_025_REPA/vae-sd"
        # elif self.mode=="muddled_0_050":
        #     self.images_dir = "/home/arsen.abzhanov/Thesis_local/REPA/ISIC_2018_muddled_0_050_REPA/images"
        #     self.features_dir = "/home/arsen.abzhanov/Thesis_local/REPA/ISIC_2018_muddled_0_050_REPA/vae-sd"
        # elif self.mode=="muddled_0_075":
        #     self.images_dir = "/home/arsen.abzhanov/Thesis_local/REPA/ISIC_2018_muddled_0_075_REPA/images"
        #     self.features_dir = "/home/arsen.abzhanov/Thesis_local/REPA/ISIC_2018_muddled_0_075_REPA/vae-sd"
        # elif self.mode=="muddled_0_1":
        #     self.images_dir = "/home/arsen.abzhanov/Thesis_local/REPA/ISIC_2018_muddled_0_1_REPA/images"
        #     self.features_dir = "/home/arsen.abzhanov/Thesis_local/REPA/ISIC_2018_muddled_0_1_REPA/vae-sd"
        # elif self.mode.startswith("busi_"):
        #     self.images_dir = "/home/arsen.abzhanov/Thesis_local/REPA/BUSI_REPA/images"
        #     self.features_dir = "/home/arsen.abzhanov/Thesis_local/REPA/BUSI_REPA/vae-sd"
        # elif self.mode=="idrid_train" or self.mode=="idrid_val":
        #     self.images_dir = "/home/arsen.abzhanov/Thesis_local/REPA/IDRID_REPA/train/images"
        #     self.features_dir = "/home/arsen.abzhanov/Thesis_local/REPA/IDRID_REPA/train/vae-sd"
        # elif self.mode=="idrid_test":
        #     self.images_dir = "/home/arsen.abzhanov/Thesis_local/REPA/IDRID_REPA/test/images"
        #     self.features_dir = "/home/arsen.abzhanov/Thesis_local/REPA/IDRID_REPA/test/vae-sd"

        BASE_DIR = "."

        MODE_TO_PATHS = {
            "test": {
                "images": f"{BASE_DIR}/ISIC_2018_REPA/test/images",
                "features": f"{BASE_DIR}/ISIC_2018_REPA/test/vae-sd"
            },
            "val": {
                "images": f"{BASE_DIR}/ISIC_2018_REPA/validation/images",
                "features": f"{BASE_DIR}/ISIC_2018_REPA/validation/vae-sd"
            },
            "train": {
                "images": os.path.join(data_dir, 'images'),
                "features": os.path.join(data_dir, 'vae-sd')
            },
            "muddled_0_025": {
                "images": f"{BASE_DIR}/ISIC_2018_muddled_0_025_REPA/images",
                "features": f"{BASE_DIR}/ISIC_2018_muddled_0_025_REPA/vae-sd"
            },
            "muddled_0_050": {
                "images": f"{BASE_DIR}/ISIC_2018_muddled_0_050_REPA/images",
                "features": f"{BASE_DIR}/ISIC_2018_muddled_0_050_REPA/vae-sd"
            },
            "muddled_0_075": {
                "images": f"{BASE_DIR}/ISIC_2018_muddled_0_075_REPA/images",
                "features": f"{BASE_DIR}/ISIC_2018_muddled_0_075_REPA/vae-sd"
            },
            "muddled_0_1": {
                "images": f"{BASE_DIR}/ISIC_2018_muddled_0_1_REPA/images",
                "features": f"{BASE_DIR}/ISIC_2018_muddled_0_1_REPA/vae-sd"
            },
            "muddled_0_15": {
                "images": f"{BASE_DIR}/ISIC_2018_muddled_0_15_REPA/images",
                "features": f"{BASE_DIR}/ISIC_2018_muddled_0_15_REPA/vae-sd"
            },
            "muddled_0_2": {
                "images": f"{BASE_DIR}/ISIC_2018_muddled_0_2_REPA/images",
                "features": f"{BASE_DIR}/ISIC_2018_muddled_0_2_REPA/vae-sd"
            },
            "busi": {
                "images": f"{BASE_DIR}/BUSI_REPA/images",
                "features": f"{BASE_DIR}/BUSI_REPA/vae-sd"
            },
            "idrid_train": {
                "images": f"{BASE_DIR}/IDRID_REPA/train/images",
                "features": f"{BASE_DIR}/IDRID_REPA/train/vae-sd"
            },
            "idrid_val": {
                "images": f"{BASE_DIR}/IDRID_REPA/train/images",
                "features": f"{BASE_DIR}/IDRID_REPA/train/vae-sd"
            },
            "idrid_test": {
                "images": f"{BASE_DIR}/IDRID_REPA/test/images",
                "features": f"{BASE_DIR}/IDRID_REPA/test/vae-sd"
            },
            "idrid_edema_train": {
                "images": f"{BASE_DIR}/IDRID_Edema_REPA/train/images",
                "features": f"{BASE_DIR}/IDRID_Edema_REPA/train/vae-sd"
            },
            "idrid_edema_val": {
                "images": f"{BASE_DIR}/IDRID_Edema_REPA/train/images",
                "features": f"{BASE_DIR}/IDRID_Edema_REPA/train/vae-sd"
            },
            "idrid_edema_test": {
                "images": f"{BASE_DIR}/IDRID_Edema_REPA/test/images",
                "features": f"{BASE_DIR}/IDRID_Edema_REPA/test/vae-sd"
            }
        }

        # Use the mapping
        mode = self.mode if not self.mode.startswith("busi_") else "busi"
        try:
            paths = MODE_TO_PATHS[mode]
            self.images_dir = paths["images"]
            self.features_dir = paths["features"]
        except KeyError:
            raise ValueError(f"Unknown mode: {self.mode}")

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

        if self.mode.startswith("busi_") or self.mode.startswith("idrid_") and self.mode!="idrid_test" and self.mode!='idrid_edema_test':
            # Get unique classes
            unique_classes = np.unique(self.labels)

            # Initialize lists to hold training and validation data
            train_indices = []
            val_indices = []
            if self.mode.startswith("busi_"):
                test_indices = []

            # Split each class
            for cls in unique_classes:
                # Get indices of all samples in the current class
                cls_indices = np.where(self.labels == cls)[0]
                
                # Split the indices into training and validation sets
                cls_train_indices, cls_val_indices = train_test_split(cls_indices, test_size=0.2, random_state=42)
                if self.mode.startswith("busi_"):
                    cls_train_indices, cls_test_indices = train_test_split(cls_train_indices, test_size=0.2, random_state=42)
                
                # Append to the respective lists
                train_indices.extend(cls_train_indices)
                val_indices.extend(cls_val_indices)
                if self.mode.startswith("busi_"):
                    test_indices.extend(cls_test_indices)

            # Convert lists to numpy arrays
            train_indices = np.array(train_indices)
            val_indices = np.array(val_indices)
            if self.mode.startswith("busi_"):
                test_indices = np.array(test_indices)

            if self.mode=="busi_train" or self.mode=="idrid_train" or self.mode=="idrid_edema_train":
                self.image_fnames = [self.image_fnames[i] for i in train_indices]
                self.feature_fnames = [self.feature_fnames[i] for i in train_indices]
                self.labels = self.labels[train_indices]
            elif self.mode == "busi_val" or self.mode=="idrid_val" or self.mode=="idrid_edema_val":
                self.image_fnames = [self.image_fnames[i] for i in val_indices]
                self.feature_fnames = [self.feature_fnames[i] for i in val_indices]
                self.labels = self.labels[val_indices]
            elif self.mode == "busi_test":
                self.image_fnames = [self.image_fnames[i] for i in test_indices]
                self.feature_fnames = [self.feature_fnames[i] for i in test_indices]
                self.labels = self.labels[test_indices]
        

        # Print the number of samples in each set
        print(f"Number of samples: {len(self.image_fnames)}")

    def _file_ext(self, fname):
        return os.path.splitext(fname)[1].lower()

    def __len__(self):
        assert len(self.image_fnames) == len(self.feature_fnames), \
            "Number of feature files and label files should be same"
        return len(self.feature_fnames)

    def __getitem__(self, idx):
        #set_seed(self.seed)
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
            #print(f"shape of image {torch.from_numpy(image).shape}")
            return self.transforms((torch.from_numpy(image), torch.from_numpy(features))), torch.tensor(self.labels[idx])
        else:
            return torch.from_numpy(image), torch.from_numpy(features), torch.tensor(self.labels[idx])