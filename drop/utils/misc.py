from typing import Callable, Generic, Iterable, List, Optional, Tuple, TypedDict, TypeVar, Union, cast, Dict, Any
import csv
from omegaconf import OmegaConf
import numpy as np
import torch
import random


def set_random_seed(random_seed: int) -> None:
    """
    Initialize random seed for random, numpy, and pytorch.
    Also make it so pytorch runs only deterministically, and
    interrupts program execution if it can't.
    Parameters
    ----
    random_seed : Optional[int]
        Random seed for the random, numpy, and pytorch libraries.
    enforce_deterministic: bool
        Enforce usage of Pytorch deterministic functions, and throw an error when
    a non-deterministic function is used.
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)


def removeOmegaConf(obj):
    try:
        return OmegaConf.to_object(obj)
    except:
        return obj

def write_dict_to_csv(file_name, write_mode, res_dict, write_header=False):
    with open(file_name, write_mode, encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[*res_dict])
        for item in [res_dict]:
            if write_header == True:
                writer.writeheader()
            if write_mode == "a":
                writer.writerow(item)

