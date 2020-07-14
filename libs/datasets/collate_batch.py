# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from utils.image_list import to_image_list

class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0, df_used=False, boundary=False):
        self.size_divisible = size_divisible
        self.df_used = df_used
        self.boundary = boundary

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = to_image_list(transposed_batch[0], self.size_divisible)
        targets = to_image_list(transposed_batch[1], self.size_divisible)
        
        dfs = to_image_list(transposed_batch[2], self.size_divisible) if self.df_used else None
        
        dist_maps = to_image_list(transposed_batch[3], self.size_divisible) if self.boundary else None

        return images, targets, dfs, dist_maps

