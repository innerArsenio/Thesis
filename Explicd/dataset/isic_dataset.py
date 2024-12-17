import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import math
import pdb 
#from Explicd.utils import GaussianLayer
import random

class SkinDataset(Dataset):

    def __init__(self, dataset_dir, mode='train', transforms=None, flag=0, debug=False, config=None, return_concept_label=False):
        self.mode = mode
        if debug:
            dataset_dir = dataset_dir + 'debug'
        print(dataset_dir)
        self.dataset_dir = dataset_dir
        self.transforms = transforms
        self.flag = flag
        self.args = config
        self.return_concept_label = return_concept_label
        
        use_orig_data=True
    
        print('start loading %s dataset'%mode)
        

        # if self.mode!="test":
        #     # training
        #     data = np.load(dataset_dir+'dataList.npy', mmap_mode='r', allow_pickle=False)
        #     label = np.load(dataset_dir+'labelList.npy', mmap_mode='r', allow_pickle=False)
        #     sincerity = label
        # else:
        #     data = np.load(dataset_dir+'test_dataList.npy', mmap_mode='r', allow_pickle=False)
        #     label = np.load(dataset_dir+'test_labelList.npy', mmap_mode='r', allow_pickle=False)
        #     sincerity = label

        if use_orig_data:
            print("using original data")
            if self.mode!="test":
                data = np.load(dataset_dir+'dataList.npy', mmap_mode='r', allow_pickle=False)
                label = np.load(dataset_dir+'labelList.npy', mmap_mode='r', allow_pickle=False)
                sincerity = np.ones_like(label)
            else:
                data = np.load(dataset_dir+'test_dataList.npy', mmap_mode='r', allow_pickle=False)
                label = np.load(dataset_dir+'test_labelList.npy', mmap_mode='r', allow_pickle=False)
                sincerity = np.ones_like(label)
            
            # Create a mapping for replacements
            replacement_mapping = {0: 4, 1: 5, 2: 1, 3: 0, 4: 2, 5: 3, 6:6}

            # Replace the elements in the array based on the mapping
            transformed_label = np.vectorize(replacement_mapping.get)(label)

            label = transformed_label.astype(np.uint8)
        else:
            print("using combo data")
            if self.mode!="test":
                # training
                data = np.load(dataset_dir+'training_combo_dataList_2.npy', mmap_mode='r', allow_pickle=False)
                label = np.load(dataset_dir+'training_combo_labelList_2.npy', mmap_mode='r', allow_pickle=False)
                sincerity = np.load(dataset_dir+'training_combo_sincerityList_2.npy', mmap_mode='r', allow_pickle=False)

                print(sincerity)
                indices_with_0 = np.where(sincerity == 0)[0]
                print(f"Total elements with sincerity == 0: {len(indices_with_0)}")
            else:
                data = np.load(dataset_dir+'test_dataList.npy', mmap_mode='r', allow_pickle=False)
                label = np.load(dataset_dir+'test_labelList.npy', mmap_mode='r', allow_pickle=False)
                sincerity = label
                

            if mode!="test":
                percentage_to_remove = float(input("Enter percentage of data with label2 == 0 to remove (0-100): "))
                # Step 3: Calculate how many elements to remove
                num_to_remove = int(len(indices_with_0) * (percentage_to_remove / 100))

                # Step 4: Randomly select the indices to remove (if any)
                indices_to_remove = np.random.choice(indices_with_0, num_to_remove, replace=False)

                # Step 5: Create a mask to exclude the selected indices
                mask = np.ones(len(sincerity), dtype=bool)
                mask[indices_to_remove] = False

                # Step 6: Apply the mask to all arrays to "remove" elements
                data = data[mask]
                label = label[mask]
                sincerity = sincerity[mask]


        
        rng = np.random.default_rng(29)
        #print(f"data.shape {data.shape}")
        shuffled_indices = rng.permutation(data.shape[0])
        
        self.dataList = data
        self.labelList = label
        self.sincerityList = sincerity
        if mode!="test":
            #self.sincerityList = sincerity
            print("uncomment")
        else:
            self.sincerityList = label

        shuffled_label = label[shuffled_indices]
        #print(f"shuffled_label {shuffled_label}")
        
        self.origin_size = [data.shape[1], data.shape[2]]
        
        index_list = np.zeros((0), dtype=np.int64)
        for i in range(7):
            print('load class %d'%i)
            num = (shuffled_label==i).sum()
            #num = num // 5
            num = num // 10
            
            index = np.where(shuffled_label == i)[0]
            test_index = index[num*flag:num*(flag+1)]
            #train_index = np.array(list(set(index) - set(test_index)))

            if self.mode == 'test':
                train_index = np.array(list(set(index)))
            else:
                train_index = np.array(list(set(index) - set(test_index)))

            #num_val = len(test_index) // 2
            if self.mode == 'train':
                index_list = np.concatenate((index_list, shuffled_indices[train_index]), axis=0)


            elif self.mode == 'val':
                #index_list = np.concatenate((index_list, shuffled_indices[test_index[:num_val]]), axis=0)
                index_list = np.concatenate((index_list, shuffled_indices[test_index]), axis=0)
            else:
                #index_list = np.concatenate((index_list, shuffled_indices[test_index[num_val:]]), axis=0)
                index_list = np.concatenate((index_list, shuffled_indices[train_index]), axis=0)

        self.index_list = index_list
        
        
        # self.concept_label_map = [
        #     [0, 0, 0, 0, 0, 0, 0], # MEL
        #     [1, 1, 1, 1, 1, 1, 0], # NV
        #     [2, 0, 2, 2, 2, 0, 1], # BCC
        #     [3, 0, 0, 3, 3, 0, 2], # AKIEC
        #     [4, 2, 1, 4, 4, 1, 3], # BKL
        #     [5, 1, 1, 5, 5, 1, 0], # DF
        #     [6, 3, 1, 6, 1, 2, 0], # VASC
        # ]

        self.concept_label_map = [
            [3, 0, 0, 3, 3, 0, 2], # AKIEC
            [2, 0, 2, 2, 2, 0, 1], # BCC
            [4, 2, 1, 4, 4, 1, 3], # BKL
            [5, 1, 1, 5, 5, 1, 0], # DF
            [0, 0, 0, 0, 0, 0, 0], # MEL
            [1, 1, 1, 1, 1, 1, 0], # NV
            [6, 3, 1, 6, 1, 2, 0], # VASC
        ]

        print('load done')
    
    def __len__(self):
        return self.index_list.shape[0]

    def __getitem__(self, idx):
        real_idx = self.index_list[idx]
        img = self.dataList[real_idx]
        label = int(self.labelList[real_idx])
        sincerity = int(self.sincerityList[real_idx])
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        if self.return_concept_label:
            return img, label, np.array(self.concept_label_map[label]), sincerity
        else:
            return img, label
   
    def random_deform(self, img):
        
        img = img.unsqueeze(0)

        dx = (torch.randn_like(img[0, 0, :, :])*2-1) * self.args.deform_scale
        dy = (torch.randn_like(img[0, 0, :, :])*2-1)  * self.args.deform_scale

        dx = self.smooth(dx.unsqueeze(0).unsqueeze(0))[0, 0, :, :].detach()
        dy = self.smooth(dy.unsqueeze(0).unsqueeze(0))[0, 0, :, :].detach()


        x_line = torch.linspace(-1, 1, steps=img.shape[2]).unsqueeze(0)
        x_line = x_line.repeat(img.shape[2], 1)
        y_line = torch.linspace(-1, 1, steps=img.shape[3]).unsqueeze(1)
        y_line = y_line.repeat(1, img.shape[3])

        x_line += dx
        y_line += dy

        grid = torch.stack((x_line, y_line), dim=2)
        grid = grid.unsqueeze(0)

        img = F.grid_sample(img, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        
    
        img = img.squeeze(0)

        return img


    def random_affine(self, img):
        
        img = img.unsqueeze(0)

        scale_x = np.random.random() * (2*self.args.scale) + 1 - self.args.scale
        scale_y = np.random.random() * (2*self.args.scale) + 1 - self.args.scale
        
        shear_x = np.random.random() * self.args.scale
        shear_y = np.random.random() * self.args.scale
        #shear_x = 0
        #shear_y = 0


        angle = np.random.randint(-self.args.angle, self.args.angle)
        angle = (angle / 180.) * math.pi

        theta_scale = torch.tensor([[scale_x, shear_x, 0],
                                    [shear_y, scale_y, 0],
                                    [0, 0, 1]]).float()
        theta_rotate = torch.tensor([[math.cos(angle), -math.sin(angle), 0],
                                    [math.sin(angle), math.cos(angle), 0],
                                    [0, 0, 1]]).float()

        theta = torch.mm(theta_scale, theta_rotate)[0:2, :]
        grid = F.affine_grid(theta.unsqueeze(0), img.size(), align_corners=True)
        
        img = F.grid_sample(img, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        img = img.squeeze(0)

        return img

