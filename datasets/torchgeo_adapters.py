from torch.utils.data import Dataset


class TgChangeDetectionAsNumpy(Dataset):
    def __init__(self, tg_dataset):
        self.tg_dataset = tg_dataset

    def _to_hwc_numpy_uint8(self, image):
      return image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

    def __getitem__(self, idx):
        sample = self.tg_dataset[idx]
        image_set = sample['image']

        image_a, image_b = image_set[0], image_set[1]
        image_a_np = self._to_hwc_numpy_uint8(image_a)
        image_b_np = self._to_hwc_numpy_uint8(image_b)

        return image_a_np, image_b_np

    def __len__(self):
        return len(self.tg_dataset)


class TgSegmentationAsNumpy(Dataset):
    def __init__(self, tg_dataset):
        self.tg_dataset = tg_dataset

    def __getitem__(self, idx):
        sample = self.tg_dataset[idx]
        image = sample['image'].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        return image

    def __len__(self):
        return len(self.tg_dataset)