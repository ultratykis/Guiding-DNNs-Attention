import numpy as np

from tqdm import tqdm
import torch
from torchvision.datasets import ImageFolder, folder


class ImageFolderWithMarkers(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=folder.default_loader, is_valid_file=None):
        super(ImageFolderWithMarkers, self).__init__(
            root, transform, target_transform, loader, is_valid_file)

        for i, s in enumerate(tqdm(self.samples, desc='>> Loading dataset...', total=len(self.samples))):
            s = list(s)
            s.append(torch.zeros([224, 224]).numpy())   # add markers
            # add markers
            s.append(torch.zeros([224, 224]).numpy())
            s.append(torch.zeros([2]).numpy())          # add gt
            s.append(torch.zeros([2]).numpy())          # add attention_peak
            # s.append(torch.zeros([224, 224]).numpy())   # add nega_click
            # s.append(torch.zeros([1]).numpy())          # add nega_click
            # s.append(i)                                 # add data id
            self.samples[i] = tuple(s)

    def __getitem__(self, index):
        path, target, markers, refined_markers, attention_peak, nega_click, data_id = self.samples[
            index]

        # path, target, neg_loss, refined_markers, attention_peak = self.samples[
        #     index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, markers, refined_markers, refined_markers, attention_peak


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


un_norm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


class SubsetSequentialSampler(torch.utils.data.Sampler):
    r"""Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)
