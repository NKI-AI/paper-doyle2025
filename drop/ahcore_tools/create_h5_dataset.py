from dlup.data.dataset import TiledWsiDataset
from drop.ahcore_tools.writers import H5FileImageWriter
import PIL
import numpy as np
import io
from drop.utils.logging import setup_logging

logger = setup_logging(log_name=__name__)
from pathlib import Path
from typing import Any, List, Tuple, Union, Optional
import imageio.v3 as iio
import numpy.typing as npt
from math import ceil, floor
import h5py
import PIL.Image
from drop.data.dataset_utils import open_slide_image
from drop.transforms.image_distortions import adjust_background_sloane

def read_mask(path: Path) -> np.ndarray:
    return iio.imread(path)[..., 0]


def _generator(dataset, quality: int | None = 80, compression: str = "JPEG"):
    for idx, sample in enumerate(dataset):
        buffered = io.BytesIO()  # what is this buffered thing?
        tile = sample["image"]
        # convert tile to RGB
        if "Sloane" in dataset.__dict__["_path"]: # or sample["path"]
            tile = adjust_background_sloane(tile)
            print(f"Adjusted background for {sample['path']}")
        if quality is not None:
            # If we just cast the PIL.Image to RGB, the alpha channel is set to black
            # which is a bit unnatural if you look in the image pyramid where it would be white in lower resolutions
            # this is why we take the following approach.
            background = PIL.Image.new("RGB", tile.size, (255, 255, 255))  # Create a white background
            background.paste(tile, mask=tile.split()[3])  # Paste the image using the alpha channel as mask
            background.convert("RGB").save(buffered, format=compression, quality=quality)
        else:
            tile.save(buffered, format=compression, quality=quality)
        # Now we have the image bytes
        coordinates = sample["coordinates"]
        array = np.frombuffer(buffered.getvalue(), dtype="uint8")
        yield [coordinates], array[np.newaxis, :]


def save_tiles(
    dataset: TiledWsiDataset,
    h5_writer: H5FileImageWriter,
    quality: int | None = 80,
):
    """
    Saves the tiles in the given image slide dataset to disk.

    Parameters
    ----------
    dataset : TiledWsiDataset
        The image slide dataset containing tiles of a single whole slide image.
    h5_writer : H5FileImageWriter
        The H5 writer to write the tiles to.
    quality : int | None
        If not None, the compression quality of the saved tiles in jpg, otherwise png

    """
    compression = "JPEG" if quality is not None else "PNG"
    generator = _generator(dataset, quality, compression)
    h5_writer.consume(generator)


def _save_thumbnail(
    dataset: TiledWsiDataset,
    wsi_mpp: float,
    tile_size: Tuple[int, int],
    mask: npt.NDArray[np.uint8],
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:

    target_mpp = max(wsi_mpp * 30, 30)
    tile_size = (
        min(30, tile_size[0] // 30),
        min(30, tile_size[1] // 30),
    )

    scaled_region_view = dataset.slide_image.get_scaled_view(dataset.slide_image.get_scaling(target_mpp))

    # Let us write the mask too.
    # not working for me
    mask_io = io.BytesIO()
    mask = PIL.Image.fromarray(mask * 255, mode="L")
    mask.save(mask_io, format="JPEG", quality=75)
    mask_arr = np.frombuffer(mask_io.getvalue(), dtype="uint8")

    thumbnail_io = io.BytesIO()
    thumbnail = dataset.slide_image.get_thumbnail(tuple(scaled_region_view.size))
    thumbnail.convert("RGB").save(thumbnail_io, format="JPEG", quality=75)  # had to add format jpeg
    thumbnail_arr = np.frombuffer(thumbnail_io.getvalue(), dtype="uint8")

    background = PIL.Image.new("RGBA", tuple(scaled_region_view.size), (255, 255, 255, 255))  # changed to PIL.Image

    overlay_io = io.BytesIO()
    for d in dataset:
        tile = d["image"]
        coords = np.array(d["coordinates"])
        box = tuple(np.array((*coords, *(coords + tile_size))).astype(int))
        background.paste(tile, box)
        # draw = ImageDraw.Draw(background)
        # draw.rectangle(box, outline="red")
    background.convert("RGB").save(overlay_io, quality=75)
    overlay_arr = np.frombuffer(overlay_io.getvalue(), dtype="uint8")

    return thumbnail_arr, mask_arr, overlay_arr


def get_wsi_size_from_target_mpp(slide_image, target_mpp: float, limit_bounds: bool = True) -> tuple[int, int]:
    scaling = slide_image.get_scaling(target_mpp)
    offset = (0, 0)
    if limit_bounds:
        offset, bounds = slide_image.slide_bounds
        offset = (int(scaling * offset[0]), int(scaling * offset[1]))
        size = int(bounds[0] * scaling), int(bounds[1] * scaling)
    else:
        size = slide_image.get_scaled_size(scaling)

    return size, offset


def vis_h5_tile(h5_path, tile_no, wsi_name, target_mpp, tile_size):
    f = h5py.File(h5_path, "r")
    binary_data = f["data"][tile_no]
    img = PIL.Image.open(io.BytesIO(binary_data))
    img = img.convert("RGB")  # image is RGBA initially
    save_path = f"/home/s.doyle/filtered_for_sat_h5_{wsi_name}_mpp{target_mpp}_{tile_size[0]}_{tile_no}.png"
    img.save(save_path)
    print(save_path)
    f.close()


def h5_tiling_pipeline(
    output_file: str,
    dataset: TiledWsiDataset,
    target_mpp: float,
    tile_size: Tuple[int, int],
    mask_path: Union[str, Path],
):
    """Tiling mode for dataset should be overflow. It only includes tiles that are masked, i.e. not background."""
    if type(dataset) == TiledWsiDataset:
        slide_image = dataset.slide_image
    else:
        slide_path = (
            dataset.path
        )  # there is an error because attribute _apply_color_profile is not defined in TiledROIsSlideImageDataset
        slide_image = open_slide_image(slide_path)
    size, offset = get_wsi_size_from_target_mpp(slide_image, target_mpp)
    h5_writer = H5FileImageWriter(
        filename=output_file,
        size=size,
        offset=offset,
        mpp=target_mpp,
        tile_size=tile_size,
        tile_overlap=(0, 0),
        num_samples=len(dataset),
        masked_indices=dataset.__dict__["masked_indices"],
        is_binary=True,
    )
    try:
        save_tiles(dataset, h5_writer, quality=80)
        print("successfully saved h5 tiles")
    except Exception as e:
        logger.error(f"Failed to store tiles: {output_file} with exception {e}")
        breakpoint()
        return
    logger.debug("Working on %s. Writing to %s", dataset.path, output_file)

    save_thumbnail = False  # does not work properly
    if save_thumbnail:
        mask = read_mask(mask_path)
        thumbnail, mask, overlay = _save_thumbnail(dataset, dataset.slide_image.mpp, tile_size, mask)
        h5_writer.add_associated_images(("thumbnail", thumbnail), ("mask", mask), ("overlay", overlay))
