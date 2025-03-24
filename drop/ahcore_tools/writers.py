# encoding: utf-8
import json
from pathlib import Path
from typing import Any, Generator, Optional, Tuple

import h5py
import numpy as np
import numpy.typing as npt
from dlup.tiling import Grid, GridOrder, TilingMode

# from ahcore.utils.io import get_logger
from drop.utils.logging import setup_logging

logger = setup_logging(log_name=__name__)


def append_grid_to_h5(h5_path, wsi_size, tile_size):
    """Appends the grid to the h5 file."""

    with h5py.File(f"{h5_path}", "a") as f:
        grid = Grid.from_tiling(
            (0, 0),
            size=wsi_size,
            tile_size=tile_size,
            tile_overlap=(0, 0),
            mode=TilingMode.overflow,
            order=GridOrder.C,
        )
        f.create_dataset(
            "grid",
            data=grid,
            dtype=int,
            compression="gzip",
        )
        # todo this seems to not be the right way to save the grid


def append_mpp_to_h5(h5_path, mpp):
    """Appends the grid to the h5 file."""

    with h5py.File(f"{h5_path}", "a") as f:
        f.create_dataset("mpp", data=mpp, dtype=float)


class H5FileImageWriter:
    """Image writer that writes tile-by-tile to h5."""

    def __init__(
        self,
        filename: Path,
        offset: tuple[int, int],
        size: tuple[int, int],
        mpp: float,
        tile_size: tuple[int, int],
        tile_overlap: tuple[int, int],
        num_samples: int,
        masked_indices: Optional[npt.NDArray[np.int_]] = None,
        is_binary: bool = False,
        progress: Optional[Any] = None,
    ) -> None:
        self._grid: Optional[Grid] = None
        self._grid_coordinates: Optional[npt.NDArray[np.int_]] = None
        self._filename: Path = filename
        self._offset: tuple[int, int] = offset
        self._size: tuple[int, int] = size
        self._mpp: float = mpp
        self._tile_size: tuple[int, int] = tile_size
        self._tile_overlap: tuple[int, int] = tile_overlap
        self._num_samples: int = num_samples
        self._masked_indices = masked_indices

        self._is_binary: bool = is_binary
        self._progress = progress

        self._data: Optional[h5py.Dataset] = None
        self._coordinates_dataset: Optional[h5py.Dataset] = None
        self._tile_indices: Optional[h5py.Dataset] = None
        self._current_index: int = 0

        self._logger = logger  # maybe not the best way, think about it
        self._logger.debug("Writing h5 to %s", self._filename)

    def init_writer(self, first_batch: Any, h5file: h5py.File) -> None:
        """Initializes the image_dataset based on the first tile."""
        batch_shape = np.asarray(first_batch).shape
        batch_dtype = np.asarray(first_batch).dtype

        self._current_index = 0

        self._coordinates_dataset = h5file.create_dataset(
            "coordinates",
            shape=(self._num_samples, 2),
            dtype=int,
            compression="gzip",
        )
        self._masked_indices = h5file.create_dataset(
            "masked_indices",
            data=self._masked_indices,
            dtype=int,
            compression="gzip",
        )
        self._target_mpp = h5file.create_dataset("mpp", data=self._mpp, dtype=float)

        # TODO: We only support a single Grid
        grid = Grid.from_tiling(
            offset=self._offset,
            size=self._size,
            tile_size=self._tile_size,
            tile_overlap=self._tile_overlap,
            mode=TilingMode.overflow,
            order=GridOrder.C,
        )
        num_tiles = len(grid)
        self._grid = grid

        self._grid = h5file.create_dataset(
            "grid",
            data=grid,
            compression="gzip",
        )

        self._tile_indices = h5file.create_dataset(
            "tile_indices",
            shape=(num_tiles,),
            dtype=int,
            compression="gzip",
        )
        # Initialize to -1, which is the default value
        self._tile_indices[:] = -1

        if not self._is_binary:
            self._data = h5file.create_dataset(
                "data",
                shape=(self._num_samples,) + batch_shape[1:],
                dtype=batch_dtype,
                compression="gzip",
                chunks=(1,) + batch_shape[1:],
            )
        else:
            dt = h5py.vlen_dtype(np.dtype("uint8"))  # Variable-length uint8 data type
            self._data = h5file.create_dataset(
                "data",
                shape=(self._num_samples,),
                dtype=dt,
                chunks=(1,),
            )

        # This only works when the mode is 'overflow' and in 'C' order.
        metadata = {
            "mpp": self._mpp,
            "dtype": str(batch_dtype),
            "shape": tuple(batch_shape[1:]),
            "size": (int(self._size[0]), int(self._size[1])),
            "num_channels": batch_shape[1],
            "num_samples": self._num_samples,
            "tile_size": tuple(self._tile_size),
            "tile_overlap": tuple(self._tile_overlap),
            "num_tiles": num_tiles,
            "grid_order": "C",
            "mode": "overflow",
            "is_binary": self._is_binary,
        }
        metadata_json = json.dumps(metadata)
        h5file.attrs["metadata"] = metadata_json

    def add_associated_images(
        self, images: tuple[tuple[str, npt.NDArray[np.uint8]], ...], description: Optional[str] = None
    ) -> None:
        """Adds associated images to the h5 file."""

        # Create a compound dataset "associated_images"
        with h5py.File(self._filename, "r+") as h5file:
            associated_images = h5file.create_group("associated_images")
            for name, image in images:
                associated_images.create_dataset(name, data=image)

            if description:
                associated_images.attrs["description"] = description

    def consume(
        self,
        batch_generator: Generator[tuple[Any, Any], None, None],  # replaced genericarray from ahcore with Any
        connection_to_parent: Optional[Any] = None,  # replaced connection from ahcore with Any
    ) -> None:
        """Consumes tiles one-by-one from a generator and writes them to the h5 file."""
        grid_counter = 0

        try:
            with h5py.File(self._filename.with_suffix(".h5.partial"), "w") as h5file:
                first_coordinates, first_batch = next(batch_generator)
                self.init_writer(first_batch, h5file)

                # Mostly for mypy
                assert self._grid, "Grid is not initialized"
                assert self._tile_indices, "Tile indices are not initialized"
                assert self._data, "Dataset is not initialized"
                assert self._coordinates_dataset, "Coordinates dataset is not initialized"

                batch_generator = self._batch_generator((first_coordinates, first_batch), batch_generator)
                # progress bar will be used if self._progress is not None
                if self._progress:
                    batch_generator = self._progress(batch_generator, total=self._num_samples)

                for coordinates, batch in batch_generator:
                    # We take a coordinate, and step through the grid until we find it.
                    # Note that this assumes that the coordinates come in C-order, so we will always hit it
                    for idx, curr_coordinates in enumerate(coordinates):
                        # As long as our current coordinates are not equal to the grid coordinates, we make a step
                        while not np.all(curr_coordinates == self._grid[grid_counter]):
                            grid_counter += 1
                        # If we find it, we set it to the index, so we can find it later on
                        # This can be tested by comparing the grid evaluated at a grid index with the tile index
                        # mapped to its coordinates
                        self._tile_indices[grid_counter] = self._current_index + idx
                        grid_counter += 1

                    batch_size = batch.shape[0]


                    self._data[self._current_index : self._current_index + batch_size] = batch
                    self._coordinates_dataset[self._current_index : self._current_index + batch_size] = coordinates
                    self._current_index += batch_size  # region_index
                    # grid_local_coordinates = (np.where(self._grid.__dict__['coordinates'][0] == self._grid[grid_counter][0])[0][0],
                    #                           np.where(self._grid.__dict__['coordinates'][1] == self._grid[grid_counter][1])[0][0])
                    # grid_local_coordinates = np.unravel_index(grid_index, self.grids[starting_index][0].size)
        except Exception as e:
            self._logger.error("Error in consumer thread for %s: %s", self._filename, e, exc_info=e)
            if connection_to_parent:
                connection_to_parent.send((False, self._filename, e))  # Send a message to the parent
        else:
            # When done writing rename the file.
            self._filename.with_suffix(".h5.partial").rename(self._filename)
        finally:
            if connection_to_parent:
                connection_to_parent.send((True, None, None))
                connection_to_parent.close()

    @staticmethod
    def _batch_generator(
        first_coordinates_batch: Any, batch_generator: Generator[Any, None, None]
    ) -> Generator[Any, None, None]:
        # We yield the first batch too so the progress bar takes the first batch also into account
        yield first_coordinates_batch
        for tile in batch_generator:
            if tile is None:
                break
            yield tile
