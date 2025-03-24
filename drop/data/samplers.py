import torch
import numpy as np
import logging
import random
from collections import defaultdict
from torch.utils.data import BatchSampler, Sampler
from torch.utils.data import WeightedRandomSampler, SubsetRandomSampler, SequentialSampler
from itertools import islice
from typing import Any, Optional, Union
import math

class SubsetRandomBatchSampler(BatchSampler):
    """Returns batches from a list of indices in random order."""

    def __init__(self, sample_df, batch_size, drop_last):
        relevant_idc = sample_df.index.astype(int).to_list()

        sampler = SubsetRandomSampler(relevant_idc)
        super().__init__(sampler=sampler, batch_size=batch_size, drop_last=drop_last)


class DataFrameSubsetWeightedRandomBatchSampler(BatchSampler):
    """Return an instance of a slide with the probability of that slide
    being selected when all instances have equal chance of being selected (proportion).
    If there are more tiles in the aperio than P1000, then it Aperio will be overrerpesented, when using the same images.

    """

    def __init__(self, sample_df, batch_size, drop_last, no_eval_per_epoch=1):
        self.group_on = "tissue_number_blockid"
        self.sampling_df = sample_df[sample_df["use"]]
        weight_per_instance, num_samples = self._get_weights(sample_df)
        samples_per_eval = int(num_samples * (1 / no_eval_per_epoch))
        sampler = WeightedRandomSampler(weight_per_instance, num_samples=samples_per_eval, replacement=True)
        super().__init__(sampler=sampler, batch_size=batch_size, drop_last=drop_last)

    def _get_weights(self, sample_df):
        num_samples = sample_df["use"].value_counts()[True]
        if "num_regions" not in sample_df.columns:
            image_counts = sample_df.groupby(self.group_on).size().reset_index(name="num_regions")
            sample_df = sample_df.merge(image_counts, on=self.group_on)
        sample_df = sample_df.assign(
            weight_per_instance=sample_df["num_regions"] / num_samples
        )
        weight_per_instance = sample_df["weight_per_instance"]
        # set the weight to 0 for the idc that we don't want to sample from
        weight_per_instance = weight_per_instance * sample_df["use"]
        return weight_per_instance.tolist(), num_samples


class DataFrameSequentialBatchSampler(BatchSampler):
    def __init__(self, sample_df, batch_size, drop_last, no_eval_per_epoch=1):
        if "use" in sample_df.columns:
            sample_df = sample_df[sample_df["use"]]
        self.sampling_df = sample_df
        self.indices = self.sampling_df.index.tolist()
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_samples = int(len(self.indices) * (1 / no_eval_per_epoch))

    def __iter__(self):
        batch = []
        for index in self.indices:
            batch.append(index)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if not self.drop_last and batch:
            yield batch

    def __len__(self):
        if self.drop_last:
            return self.num_samples // self.batch_size
        else:
            return (self.num_samples + self.batch_size - 1) // self.batch_size



class NewImageBatchSampler(Sampler):
    """
    This sampler creates batches by sampling from 'm' images per batch and subsampling 'n' samples per image.
    It behaves as a sequential sampler if 'shuffle' is False and as a random sampler if 'shuffle' is True
    (shuffling occurs every epoch).

    Args:
        sample_df (pd.DataFrame): DataFrame containing image data.
        no_eval_per_epoch (int): Number of evaluations per epoch. Default is 1.
        m_images (int): Number of images per batch. Default is 1.
        shuffle (bool): Whether to shuffle the data. Default is True.
        n_samples (int or None): Number of samples to include in each batch. Default is None.
    """

    def __init__(
        self,
        sample_df,
        no_eval_per_epoch=1,
        shuffle=True,
        m_images=1,
        n_samples: Optional[Union[int, None]] = None,
    ):
        if "use" in sample_df.columns:
            sample_df = sample_df[sample_df["use"]]
        self.sampling_df = sample_df
        self.no_eval_per_epoch = no_eval_per_epoch
        self.shuffle = shuffle
        self.m_images = m_images
        self.n_samples = n_samples

        self.unique_images = sample_df["imageName"].unique()
        self.num_samples = math.ceil(len(self.unique_images)/ self.m_images)
        self.slide_sampler = SequentialSampler(self.unique_images)
        # Shuffle the unique images for the first epoch
        self.shuffle_images()
        self.counter = 0

    def shuffle_images(self):
        # Shuffle the unique images in place
        self.shuffled_img_idc = torch.randperm(len(self.unique_images))
        self.shuffled_images_list = self.unique_images[self.shuffled_img_idc]

    def __iter__(self):
        if self.shuffle and self.counter == len(self.unique_images):
            self.shuffle_images()
            self.counter = 0

        for image_idx in self.slide_sampler:
            batch = []
            complete_batch = True
            for mi in range(self.m_images):
                indices = self.sampling_df[
                    self.sampling_df["imageName"] == self.shuffled_images_list[image_idx]
                ].index.astype(int)
                indices = indices.tolist()
                if self.n_samples is not None and len(indices) > self.n_samples:
                    indices = random.sample(indices, self.n_samples)
                self.counter += self.m_images
                # shuffles indices in place
                random.shuffle(indices)
                batch.extend(indices)

            if mi == self.m_images - 1:
                # should also return batch if it doesn't have n_samples * m_images
                if self.n_samples is not None:
                    complete_batch = len(batch) == self.m_images * self.n_samples  # sanity check # not sure how to hanfle exceptions
                if complete_batch:
                    yield batch
                else:
                    yield batch

    def __len__(self):
        return int(self.num_samples * (1 / self.no_eval_per_epoch))

