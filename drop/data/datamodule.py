from typing import List, Optional, Tuple, Dict, Any, TypeVar, Union
DataFrame = TypeVar("pandas.core.frame.DataFrame")
import logging
from torch.utils.data import DataLoader
from pytorch_lightning.core.datamodule import LightningDataModule
from torch.utils.data.dataloader import default_collate


def resolve_name(name):
    if name is None:
        return None
    components = name.split(".")
    obj = __import__(components[0])
    for component in components[1:]:
        obj = getattr(obj, component)
    return obj


def deepmil_collate_fn(batch):
    """
    Keys that are removed:
     ["grid_local_coordinates", "coordinates", "scaling_factor", "wsi_mpp", "mpp"]
    Grid_local_coordinates and coordinates are stored as a tuple for x and y coordinates.
    """
    batch = default_collate(batch)
    no_collapse_keys = ["x", "region_index", "sample_index"]
    collapse_keys = ["y", "imageID", "imageName", "subdir"]
    keep_keys = no_collapse_keys + collapse_keys
    new_batch = {k: (v[0:1] if k in collapse_keys else v) for k, v in batch.items() if k in keep_keys}
    return new_batch


class DataModule(LightningDataModule):
    def __init__(
        self,
        dataset: Any,
        batch_size: int,
        num_workers: int,
        sampler: Any,
        sample_col: str,
        collate_fn: Optional[Union[str, None]] = None,
        fold: Optional[int] = None,
        folds_df: Optional[DataFrame] = None,
    ):
        super().__init__()
        self.ds = dataset
        self.num_workers = num_workers
        self.fold = str(fold)
        self.folds_df = folds_df
        self.batch_size = batch_size
        if self.batch_size is None:
            logging.warning("Batch size is None.")
        self.sampler = sampler
        self.sample_col = sample_col
        self.collate_fn = resolve_name(collate_fn)

        # self.save_hyperparameters() #do not do this, mlflow logger crashes
        self._train_fold_df = None
        self._val_fold_df = None
        self._test_df = None
        self._inference_df = None

    def setup(self, stage: str):
        """
        Fit:
        Assign train/val datasets for use in dataloaders.
        We get all the samples idx and split them based on label and slide (with the cv-splits method).
        We can then sample from the train or the val dataset (which are the same dataset except that they apply different
        transforms to the data_proc) based on the train_idc and val_idc.
        Test:
        Assign test dataset for use in dataloader.
        None:
        Assign inference dataset for use in dataloader.

        """
        type_dataset = self.ds.func.__module__
        if type_dataset not in ["drop.data.cls.mil_region_dataset", "drop.data.drop_dataset"]:
            raise ValueError("Dataset type not recognized.")

        if stage == "fit":
            logging.info(f"Building {type_dataset}")
            self.fit_dataset = self.ds(split="train")
            logging.info("Fit dataset created")

            if type_dataset == "drop.data.cls.mil_region_dataset":
                self._train_fold_df = self.folds_df.loc[self.folds_df[self.fold] == "train"][
                    [self.sample_col, self.fold]
                ]  # each row is a region
                self._val_fold_df = self.folds_df.loc[self.folds_df[self.fold] == "val"]

            elif type_dataset == "drop.data.drop_dataset":
                names_train_fold = (
                    self.folds_df.loc[self.folds_df[self.fold] == "train"][self.sample_col].unique().tolist()
                )
                names_val_fold = (
                    self.folds_df.loc[self.folds_df[self.fold] == "val"][self.sample_col].unique().tolist()
                )
                assert len(set(names_train_fold).intersection(set(names_val_fold))) == 0

                fit_df = self.fit_dataset.sample_df  # is expanded by num_regions
                train_fold_df = fit_df.copy()
                train_cond = fit_df[self.sample_col].isin(names_train_fold)
                self.train_fold_sampling_df = train_fold_df.assign(use=train_cond)

                val_fold_df = fit_df.copy()
                val_cond = fit_df[self.sample_col].isin(names_val_fold)
                self.val_fold_sampling_df = val_fold_df.assign(use=val_cond)

        if stage == "train_without_val":
            type_dataset = self.ds.func.__module__
            if type_dataset == "drop.data.tile_dataset":
                logging.info(f"Building {type_dataset}")
                self.fit_dataset = self.ds(split="train")
                logging.info(f"Built whole fit train dataset")
                self.train_fold_sampling_df = self.fit_dataset.sample_df
                self.train_fold_sampling_df = self.train_fold_sampling_df.assign(use=True)

        if stage == "test":
            logging.info(f"Building {type_dataset}")
            self.test_dataset = self.ds(split="test")

            logging.info(f"Built test dataset")
            # the benefit here is that it takes into consideration the slides that could not be read and also the slides
            # (they have 0 regions)
            self._test_df = self.test_dataset.sample_df # expanded by num_regions

        if (stage is None) or (stage == "inference"):
            logging.info(f"Building {type_dataset}")
            self.inference_dataset = self.ds(split=None)
            logging.info(f"Built test dataset")
            # the benefit here is that it takes into consideration the slides that could not be read and also the slides
            # (they have 0 regions)
            self._inference_df = self.inference_dataset.sample_df

    def train_dataloader(self):
        sampler = self.sampler.train(self.train_fold_sampling_df)
        return DataLoader(
            self.fit_dataset,
            num_workers=self.num_workers,
            pin_memory=False,
            batch_sampler=sampler,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        sampler = self.sampler.no_train(self.val_fold_sampling_df)
        return DataLoader(
            self.fit_dataset,
            num_workers=self.num_workers,
            pin_memory=False,
            batch_sampler=sampler,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        sampler = self.sampler.no_train(self._test_df)
        return DataLoader(
            self.test_dataset,
            num_workers=self.num_workers,
            pin_memory=False,
            batch_sampler=sampler,
            collate_fn=self.collate_fn,
        )

    def predict_dataloader(self):
        sampler = self.sampler.no_train(self._inference_df)
        return DataLoader(
            self.inference_dataset,
            num_workers=self.num_workers,
            pin_memory=False,
            batch_sampler=sampler,
            collate_fn=self.collate_fn,
        )
