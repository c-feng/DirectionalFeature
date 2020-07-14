import torch
from torch import Tensor
from typing import List, Set, Iterable
from utils.utils_loss import class2one_hot

def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())

def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)

def simplex(t: Tensor, axis=1) -> bool:
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)

def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])


class SurfaceLoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, dist_maps: Tensor, gts: Tensor) -> Tensor:
        assert simplex(probs)
        assert not one_hot(dist_maps)
        
        dist_maps = dist_maps.to(probs.device)

        pc = probs[:, self.idc, ...]
        dc = dist_maps[:, self.idc, ...]
        
        multipled = torch.einsum("bcwh,bcwh->bcwh", pc, dc)

        # gc = class2one_hot(gts)[:, self.idc, ...]
        # multipled = torch.einsum("bcwh,bcwh->bcwh", pc - gc, dc)

        loss = multipled.mean()

        return loss