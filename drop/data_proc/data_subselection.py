""" From Available Slides Enable Subselection of slides according to A Strategy"""
from typing import Callable, Generic, Iterable, List, Optional, Tuple, TypedDict, TypeVar, Union, cast, Dict, Any
import random
import math
import numpy as np

import pandas as pd

# Important! This also subsamples test set
class SubSelector:
    """Creates a selection of available slides."""

    def __init__(self):
        pass

    def __call__(self, sel_slides, command):
        self.sel_items = sel_slides
        # warning This method returns an empty list if there are no items sel_items

        if not isinstance(self.sel_items, (list, pd.core.series.Series, np.ndarray)):
            return []

        if command.strategy == "all":
            return sel_slides

        elif command.strategy == "first_n_samples":
            items = self.sel_items[: command.value]

        elif command.strategy == "last_n_samples":
            items = self.sel_items[-command.value :]

        elif command.strategy == "every_n_samples":
            items = self.sel_items[:: command.value]

        elif command.strategy == "random_subsampling":
            items = self.use_random_selection(command.value)

        elif command.strategy == "use_local_slides":
            items = self.use_slide_selection(command.local_slides)

        else:
            raise ValueError

        return items

    def use_slide_selection(self, slide_selection: List[str]):
        return [slide for slide in self.sel_items if slide in slide_selection]

    def use_random_selection(self, n: int):
        return random.choices(self.sel_items, k=n)
