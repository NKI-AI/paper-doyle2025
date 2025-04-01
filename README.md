# DCIS Risk and Outcome Prediction

This repository is dedicated to predicting the risk of invasive breast cancer (IBC) recurrence from ductal carcinoma in situ (DCIS) histopathology images. The project is part of the PRECISION grant, a collaboration between medical institutions in the Netherlands, UK, and the US.

## Installation & Setup

### Prerequisites

Ensure the following dependencies are installed:

- Python (>=3.10)
- Conda
- OpenSlide, libvips, and pixman (install using Spack if necessary)
- GCC and G++ (version between 5.0 and 10.0)

To install the required GCC version:

```bash
sudo apt-get install gcc-9 g++-9
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 40
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 40
sudo update-alternatives --config g++
sudo update-alternatives --config gcc
```

### Setting Up the Environment

Clone the repository and set up the Conda environment:

```bash
git clone <repo_url>
cd DROP
conda env create -f env.yml  # or: conda create -n new_env python=3.10
conda activate new_env
python -m pip install -e .[dev]  # 'dev' for development mode
```

Load Spack modules and activate the Conda environment:

```bash
. /sw/spack/share/spack/setup-env.sh
```

## Running Experiments

To run an experiment:

```bash
python trainer.py task=task_folder/run_script.sh
```

## Data Structure

The repository requires data to be organized as follows:

### Raw Images & Metadata

```
orig_data_dir/
└── dataset_name/
    ├── subdir/
    │   ├── images/
    │   │   ├── img1.ext
    │   │   ├── img2.ext
    │   ├── metadata/
    │       ├── slidescore_mapping_fn
    │       ├── original_metadata_file
```

### Processed Data

```
derived_data_dir/
└── dataset_name/
    ├── subdir/
    │   ├── h5_embeddings/
    │   │   ├── embeddings.h5
    │   ├── h5_images/
    │   │   ├── images.h5
    │   ├── cached_datasets/
    │   │   ├── dataset.pkl
    │   ├── tiff_background_masks/
    │   │   ├── mask.tiff
```

### Project Data

```
project_data_dir/
└── dataset/
    ├── matched_metadata_fn
    ├── server_id_slide_mapping_fn
    ├── metadata_fn
    ├── subdir/
    │   ├── masks/
    │   │   ├── mask.tiff
    ├── bbox_predictions/
    │   ├── img/
    │   │   ├── subdir/
    │   │       ├── regions.json
    │   │       ├── boxes*.csv
    │   │       ├── tile_bounds.csv
    ├── task/
    │   ├── task_data_fn
    │   ├── cv_split_fn
```

### General Project Files & Output

```
project_dir/
├── containers/
│   ├── singularity_image
├── logs_task/
│   ├── experiment_name/
│   │   ├── split-i/
│   │       ├── checkpoints/
│   │       ├── lightning_logs/  # for TensorBoard
├── ssl_d2_ckpt_dir/
│   ├── ssl_method/
│       ├── ssl_model_ep*.torch  # Torch format for classification
│       ├── ssl_model_ep*_d2.torch  # Detectron format for detection
```

## Metadata File Creation

For Precision\_NKI\_89\_05:

1. Save Excel metadata files as CSV (using only the main sheet).
2. Run:
   ```bash
   python data_proc/meta_data/make_project_data_csv.py --task make_precision_nki_meta_data_file
   ```
3. Create metadata CSV with split:
   ```bash
   python data_analysis/meta_data_check/make_project_data_csv.py --task create_split
   ```

## Model Preparation

Classification models must be in torchvision format and can be converted using scripts in `model_weight_conversion`.

## Configuration with HYDRA

All parameters can be merged with default configurations, allowing flexible overrides. Modify configuration files as needed.

---

For more details, refer to the project's documentation or contact the maintainers.
