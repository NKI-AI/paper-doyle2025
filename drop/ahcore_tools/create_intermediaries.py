import torch
import numpy as np
import h5py
from torch.utils.data import DataLoader
from typing import Any, Optional
from pathlib import Path
import os


def store_model_features_h5(features, region_index, file_path):

    with h5py.File(file_path, "a") as file:
        # Create or append to the dataset for features
        if "features" in file:
            dataset_f = file["features"]
            dataset_f.resize((dataset_f.shape[0] + features.shape[0]), axis=0)
            dataset_f[-features.shape[0] :] = features
        else:
            dataset_f = file.create_dataset(
                "features",
                data=features,
                shape=features.shape,
                dtype=np.float32,
                chunks=features.shape,
                maxshape=(None, features.shape[1]),
            )
        # Create or append to the dataset for region index
        if "region_index" in file:
            dataset = file["region_index"]
            dataset.resize((dataset.shape[0] + region_index.shape[0],))
            dataset[-region_index.shape[0] :] = region_index
        else:
            dataset = file.create_dataset(
                "region_index",
                data=region_index,
                shape=region_index.shape,
                dtype="int32",
                chunks=(1,),
                maxshape=(None,),
            )
    return features


def save_embeddings(model: Any, datamodule: Any):

    """When creating embeddings make sure: eval_per_epoch =1 , using sequential sampler, batch size = 1, shuffle = False.
    Using inference dataset.
     Make sure that the first wsi embeddings are saved"""
    datamodule.setup("inference")
    relevant_data_df = datamodule.inference_dataset.image_df
    dl = DataLoader(
        datamodule.inference_dataset, batch_size=1, shuffle=False
    )  # best to run this with BS=1, in case image changes
    # Check if CUDA (GPU) is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)
    embed_wsi = []
    region_index_wsi = []
    id_col = "imageName"
    # Set the slide id to the first slide
    current_slide_id = datamodule.inference_dataset[0][id_col]
    num_regions = relevant_data_df.loc[relevant_data_df[id_col] == current_slide_id]["num_regions"].item()

    # Check that none of the images for which we create embeddings already exist
    # change store_model_features_h5 so that appending is not supported, then this check is not needed
    for slide in relevant_data_df[id_col].unique():
        embed_path = relevant_data_df.loc[relevant_data_df[id_col] == slide]["embed_path"].item()
        if embed_path.exists():
            breakpoint()

    for idx, batch in enumerate(dl):
        img = batch["x"].to(device)
        slide_id = batch[id_col].tolist()[0] if type(batch[id_col]) == torch.Tensor else batch[id_col][0]
        region_index = batch["region_index"].numpy()
        print(region_index, slide_id)
        embed = model.forward(img)
        embed = embed.detach().cpu().numpy()
        if slide_id != current_slide_id or idx == len(dl) - 1:
            if idx == len(dl) - 1:
                embed_wsi.append(embed)
                region_index_wsi.append(region_index)
            if len(embed_wsi) != num_regions:
                breakpoint()
            embed_wsi = np.vstack(embed_wsi)
            region_index_wsi = np.hstack(region_index_wsi)
            embed_path = relevant_data_df.loc[relevant_data_df[id_col] == current_slide_id]["embed_path"].item()
            store_model_features_h5(embed_wsi, region_index_wsi, embed_path)
            print(f"Finished slide {current_slide_id}. Now Saving embeddings for slide {slide_id}")
            current_slide_id = slide_id
            num_regions = relevant_data_df.loc[relevant_data_df[id_col] == slide_id]["num_regions"].item()
            embed_wsi = [embed]
            region_index_wsi = [region_index]
        else:
            region_index_wsi.append(region_index)
            embed_wsi.append(embed)
    # Check that all embeddings have been saved
    index_list = []
    file_list = os.listdir(Path(embed_path).parent)
    for idx, slide_name in relevant_data_df["imageName"].iteritems():
        # check if a file containin that slide id already exists
        file_exists = any(slide_name in filename for filename in file_list)
        if not file_exists:
            index_list.append(idx)
    print(f"Missing embeddings for {len(index_list)} slides")
