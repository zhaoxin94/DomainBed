import random
from math import sqrt
import numpy as np

from torch.utils.data import ConcatDataset, Dataset
from torchvision import transforms


class DatasetAll_FDA(Dataset):
    """
    Combine Seperated Datasets
    """
    def __init__(self, data_list, alpha=1.0):

        self.data = ConcatDataset(data_list)

        self.pre_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(), lambda x: np.asarray(x)
        ])
        self.post_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.alpha = alpha

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]

        # randomly sample an item from the dataset
        img_s, _ = self._sample_item()

        # do pre_transform before FDA
        img = self.pre_transform(img)
        img_s = self.pre_transform(img_s)

        # FDA
        img_mix = self._colorful_spectrum_mix(img, img_s, self.alpha)

        # do post_transform after FDA
        img = self.post_transform(img)
        img_mix = self.post_transform(img_mix)

        img = [img, img_mix]
        label = [label, label]

        return img, label

    def _colorful_spectrum_mix(self, img1, img2, alpha, ratio=1.0):
        """Input image size: ndarray of [H, W, C]"""
        lam = np.random.uniform(0, alpha)

        assert img1.shape == img2.shape
        h, w, c = img1.shape
        h_crop = int(h * sqrt(ratio))
        w_crop = int(w * sqrt(ratio))
        h_start = h // 2 - h_crop // 2
        w_start = w // 2 - w_crop // 2

        img1_fft = np.fft.fft2(img1, axes=(0, 1))
        img2_fft = np.fft.fft2(img2, axes=(0, 1))
        img1_abs, img1_pha = np.abs(img1_fft), np.angle(img1_fft)
        img2_abs, img2_pha = np.abs(img2_fft), np.angle(img2_fft)

        img1_abs = np.fft.fftshift(img1_abs, axes=(0, 1))
        img2_abs = np.fft.fftshift(img2_abs, axes=(0, 1))

        img1_abs_ = np.copy(img1_abs)
        img2_abs_ = np.copy(img2_abs)
        img1_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
            lam * img2_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img1_abs_[
                                                                                            h_start:h_start + h_crop,
                                                                                            w_start:w_start + w_crop]

        img1_abs = np.fft.ifftshift(img1_abs, axes=(0, 1))
        img2_abs = np.fft.ifftshift(img2_abs, axes=(0, 1))

        img21 = img1_abs * (np.e**(1j * img1_pha))
        img21 = np.real(np.fft.ifft2(img21, axes=(0, 1)))

        img21 = np.uint8(np.clip(img21, 0, 255))

        return img21

    def _sample_item(self):
        idxs = list(range(len(self.data)))
        selected_idx = random.sample(idxs, 1)[0]
        return self.data[selected_idx]


class DatasetAll(Dataset):
    """
    Combine Seperated Datasets
    """
    def __init__(self, data_list):
        self.data = ConcatDataset(data_list)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
