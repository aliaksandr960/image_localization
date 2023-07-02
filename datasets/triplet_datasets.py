import hashlib
import cv2
import numpy as np

import albumentations as A

from torch.utils.data import Dataset

from triplet_augmentation import GEOMETRIC_DOUBLE, GEOMETRIC_SINGLE, FINE_SINGLE
from triplet_augmentation import COLOR_DOUBLE, COLOR_SINGLE
from triplet_augmentation import RANDOM_CROP_SINGLE, RANDOM_CROP_DOUBLE
from triplet_augmentation import make_no_aug


class TripletDataset(Dataset):
    def __init__(self, np_dataset, patch_size, index_list=None, resize=None):
        pass

    def __getitem__(self, index):
        raise NotImplemented()
    
    def __len__(self):
        raise NotImplemented()


class GlobalTripletRandomDataset(Dataset):
    def __init__(self, np_dataset, patch_size, index_list=None, resize=None,
                 augmentations=make_no_aug()):
        self.np_dataset = np_dataset
        self.patch_size = patch_size

        if index_list is None:
            index_list = [i for i in range(len(np_dataset))]
        self.index_list = index_list

        self.augmentations = augmentations
        self.resize = resize
        
    def __getitem__(self, index):
        i = self.index_list[index]
        anchor, positive = self.np_dataset[i]
        
        negative_indexes = [j for j in self.index_list if j !=i]
        negative_i = np.random.choice(negative_indexes)
        negative_a, negative_b = self.np_dataset[negative_i]
        negative = [negative_a, negative_b][np.random.choice([0, 1])]
        
        # Augumentation
        aug = self.augmentations # Just a shortcut
        
        # Anchor and positive should be croppted and transformed same way
        sample = aug[GEOMETRIC_DOUBLE](image=anchor, positive=positive)
        anchor, positive = sample['image'], sample['positive']
        
        # Negative could have any other transformation
        sample = aug[GEOMETRIC_SINGLE](image=negative)
        negative = sample['image']

        # Anchor image could have any color
        sample = aug[COLOR_SINGLE](image=anchor)
        anchor = sample['image']
        
        # Positive and negative should have the same color
        sample = aug[COLOR_DOUBLE](image=positive, negative=negative)
        positive, negative = sample['image'], sample['negative']
        
        # Fine augmentation
        sample = aug[FINE_SINGLE](image=anchor)
        anchor = sample['image']
        sample = aug[FINE_SINGLE](image=positive)
        positive = sample['image']
        sample = aug[FINE_SINGLE](image=negative)
        negative = sample['image']
        
        # Random cropping 
        sample = aug[RANDOM_CROP_DOUBLE](image=anchor, positive=positive)
        anchor, positive = sample['image'], sample['positive']
        sample = aug[RANDOM_CROP_SINGLE](image=negative)
        negative = sample['image']

        
        anchor, positive, negative = [cv2.resize(i, self.patch_size) for i in [anchor, positive, negative]]
        if self.resize is not None:
            anchor, positive, negative = [cv2.resize(i, self.resize) for i in [anchor, positive, negative]]
        return anchor, positive, negative
    
    def __len__(self):
        return len(self.index_list)


class GlobalTripletStaticDataset(Dataset):
    def __init__(self, np_dataset, patch_size, index_list=None, resize=None):
        
        self.np_dataset = np_dataset
        if index_list is None:
            index_list = [i for i in range(len(np_dataset))]
        self.patch_size = patch_size
        self.resize = resize
        
        # Make pseudorandom ordering
        key_f = lambda x: hashlib.md5(str(x).encode()).hexdigest()
        self.index_list = sorted(self.index_list, key=key_f)
        
        self.crop = A.Compose([A.CenterCrop(height=patch_size[0], width=patch_size[1],
                                            always_apply=True),],
                              additional_targets={'positive': 'image',
                                                  'negative': 'image'})
        
    def __getitem__(self, index):
        i = self.index_list[index]
        anchor, positive = self.np_dataset[i]

        negative_i = self.index_list[index -1]
        _, negative = self.np_dataset[negative_i]
        
        sample = self.crop(image=anchor, positive=positive, negative=negative)
        anchor, positive, negative = sample['image'], sample['positive'], sample['negative']
        
        anchor, positive, negative = [cv2.resize(i, self.patch_size) for i in [anchor, positive, negative]]
        if self.resize is not None:
            anchor, positive, negative = [cv2.resize(i, self.resize) for i in [anchor, positive, negative]]
        return anchor, positive, negative
    
    def __len__(self):
        return len(self.index_list)


class LocalTripletRandomDataset(Dataset):
    def __init__(self, np_dataset, patch_size, index_list=None, resize=None,
                 margin=64, augmentations=make_no_aug()):
        self.np_dataset = np_dataset

        if index_list is None:
            index_list = [i for i in range(len(np_dataset))]

        self.augmentations = augmentations
        self.patch_size = patch_size
        self.resize = resize
        self.margin = margin
        
    def __getitem__(self, index):
        i = self.index_list[index]
        anchor, positive = self.np_dataset[i]
        
        # Augumentation
        aug = self.augmentations # Just a shortcut
        
        # Anchor and positive should be croppted and transformed same way
        sample = aug[GEOMETRIC_DOUBLE](image=anchor, positive=positive)
        anchor, positive = sample['image'], sample['positive']

        # Anchor image could have any color
        sample = aug[COLOR_SINGLE](image=anchor)
        anchor = sample['image']
        
        # Fine augmentation
        sample = aug[FINE_SINGLE](image=anchor)
        anchor = sample['image']
        sample = aug[FINE_SINGLE](image=positive)
        positive = sample['image']

        # Random cropping
        image_h, image_w, _ = anchor.shape
        patch_h, patch_w = self.patch_size
        start_h, start_w = 0, 0
        end_h, end_w = image_h - patch_h, image_w - patch_w
        h1, w1 = int(np.random.uniform(start_h, end_h)), int(np.random.uniform(start_w, end_w))
        h2, w2 = h1 + patch_h, w1 + patch_w
        anchor_crop = anchor[h1:h2, w1:w2, :]
        positive_crop = positive[h1:h2, w1:w2, :]
        
        not_avilable_h = [i for i in range(h1 - self.margin, h1 + self.margin)]
        available_h = [i for i in range(start_h, end_h) if i not in not_avilable_h]
        nh1 = np.random.choice(available_h)
        
        not_avilable_w = [i for i in range(w1 - self.margin, w1 + self.margin)]
        available_w = [i for i in range(start_w, end_w) if i not in not_avilable_w]
        nw1 = np.random.choice(available_w)
        
        nh2, nw2 = nh1 + patch_h, nw1 + patch_w
        negative_crop = positive[nh1:nh2, nw1:nw2, :]
        
        # Positive and negative should have the same color
        sample = aug[COLOR_DOUBLE](image=positive_crop, negative=negative_crop)
        positive_crop, negative_crop = sample['image'], sample['negative']
        
        anchor, positive, negative = [cv2.resize(i, self.patch_size) for i in [anchor, positive, negative]]
        if self.resize is not None:
            anchor_crop = cv2.resize(anchor_crop, self.resize)
            positive_crop = cv2.resize(positive_crop, self.resize)
            negative_crop = cv2.resize(negative_crop, self.resize)
        return anchor_crop, positive_crop, negative_crop
    
    def __len__(self):
        return len(self.index_list)


class LocalTripletStaticDataset(Dataset):
    def __init__(self, np_dataset, patch_size, index_list=None, resize=None,
                  max_size=(1024, 1024), margin=64):
        self.np_dataset = np_dataset

        if index_list is None:
            index_list = [i for i in range(len(np_dataset))]

        self.patch_size = patch_size
        self.resize = resize
        self.margin = margin
        
        self.max_size = max_size
        max_h, max_w = self.max_size
        h_index_list = [i for i in range(max_h)]
        key_hf = lambda x: hashlib.md5(('h' + str(x)).encode()).hexdigest()
        self.h_index_list = sorted(h_index_list, key=key_hf)
        
        w_index_list = [i for i in range(max_w)]
        key_wf = lambda x: hashlib.md5(('w' + str(x)).encode()).hexdigest()
        self.w_index_list = sorted(w_index_list, key=key_wf)
        
    def __getitem__(self, index):
        i = self.index_list[index]
        anchor, positive = self.np_dataset[i]
        
        image_h, image_w, _ = anchor.shape
        patch_h, patch_w = self.patch_size
        ch, cw = image_h // 2, image_w // 2
        h1, w1 = ch - (patch_h // 2), cw - (patch_w // 2)
        h2, w2 = h1 + patch_h, w1 + patch_w
        anchor_crop = anchor[h1:h2, w1:w2, :]
        positive_crop = positive[h1:h2, w1:w2, :]

        # Random cropping
        start_h, start_w = 0, 0
        end_h, end_w = image_h - patch_h, image_w - patch_w
        
        not_avilable_h = [i for i in range(h1 - self.margin, h1 + self.margin)]
        available_h = [i for i in range(start_h, end_h) if i not in not_avilable_h]
        available_h_count = len(available_h)
        for c in self.h_index_list:
            if c in not_avilable_w:
                continue
            if c < available_h_count:
                nh1 = c
                break
                
        not_avilable_w = [i for i in range(w1 - self.margin, w1 + self.margin)]
        available_w = [i for i in range(start_w, end_w) if i not in not_avilable_h]
        available_w_count = len(available_w)
        for c in self.w_index_list:
            if c in not_avilable_w:
                continue
            if c < available_w_count:
                nw1 = c
                break
                
        nh2, nw2 = nh1 + patch_h, nw1 + patch_w
        negative_crop = positive[nh1:nh2, nw1:nw2, :]
        
        anchor, positive, negative = [cv2.resize(i, self.patch_size) for i in [anchor, positive, negative]]
        if self.resize is not None:
            anchor_crop = cv2.resize(anchor_crop, self.resize)
            positive_crop = cv2.resize(positive_crop, self.resize)
            negative_crop = cv2.resize(negative_crop, self.resize)
        return anchor_crop, positive_crop, negative_crop
    
    def __len__(self):
        return len(self.index_list)