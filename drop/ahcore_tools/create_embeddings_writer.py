import numpy.typing as npt
from drop.utils.logging import setup_logging

logger = setup_logging(log_name=__name__)
from pathlib import Path
import h5py
import numpy as np
from typing import Any, Generator, Optional, Tuple


class H5FileImageWriter:
    """Image writer that writes embeddings to h5. Is just a rough start, not tested yet or finished."""

    def __init__(
        self,
        filename: Path,
        num_tiles: int,
        masked_indices: Optional[npt.NDArray[np.int_]] = None,
    ) -> None:

        self._filename: Path = filename
        self.num_tiles: int = num_tiles
        self._data: Optional[h5py.Dataset] = None

        self._current_index: int = 0
        self._logger = logger  # maybe not the best way, think about it
        self._logger.debug("Writing h5 to %s", self._filename)

    def init_writer(self, first_batch: Any, h5file: h5py.File) -> None:
        """Initializes the image_dataset based on the first tile."""
        batch_shape = np.asarray(first_batch).shape
        batch_dtype = np.asarray(first_batch).dtype

        self._current_index = 0  # unnecessary?
        self._target_mpp = h5file.create_dataset("mpp", data=self._mpp, dtype=float)

        self._region_indices = h5file.create_dataset(
            "region_indices",
            shape=(self.num_tiles,),
            dtype=int,
            compression="gzip",
        )
        # Initialize to -1, which is the default value
        self._tile_indices[:] = -1

        self._data = h5file.create_dataset(
            "data",
            shape=(self.num_tiles,),
            dtype=batch_dtype,
            compression="gzip",
            chunks=(1,) + batch_shape[1:],
        )

    def consume(
        self,
        features: Any,
        region_idc: Any,
        connection_to_parent: Optional[Any] = None,  # replaced connection from ahcore with Any
    ) -> None:
        """Consumes tiles one-by-one from a generator and writes them to the h5 file."""
        batch_size = features.shape[0]
        self._data[self._current_index : self._current_index + batch_size] = features
        self._region_indices[self._current_index : self._current_index + batch_size] = region_idc
        self._current_index += batch_size  # region_index
        if self._current_index == self.num_tiles:
            self._filename.with_suffix(".h5.partial").rename(self._filename)
