# DCIS Risk and Outcome Prediction

Project on IBC recurrence prediction from DCIS histopathology images.
This project falls under the PRECISION grant, which sees a collaboration
between medical institutes from the Netherlands, UK, and the US.

### Usage of DROP 
- Start screen/tmux session. 
- Load openslide, libvips and pixman (with spack).
- Build conda env if necessary.
- Activate conda env (or run singularity image- not further supported).
- Run experiment with 
```bash
python trainer.py task=task_folder/run_script.sh
```
### Setup of DROP

### Usage in conda environment
Setup DROP repo in conda environment.

```bash
cd ./DROP
conda env create -f env.yml # conda create -n new_env python=3.10

python -m pip install -e . [dev]  #dev for development mode
```
Make sure that gcc and g++ version is between 5.0 and 10.0. 
Install gcc and g++ if correct version is not present.
```bash
sudo apt-get install gcc-9
sudo apt-get install g++-9
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 40
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 40
#Chose alternative version 9 using the following command 
sudo update-alternatives --config g++ 
sudo update-alternatives --config gcc
```
Load spack modules and activate conda env.
Make sure to set the conda_env name correctly in setup-env.sh
```bash
. /sw/spack/share/spack/setup-env.sh
```


### Usage with singularity image
**This is not further supported.** 

The singularity image is built from the dockerfile in the container folder.
The dockerfile is built from the setup.cfg file.
**Todo:**
Add the packages from setup.cfg to dockerfile and build singularity container. Or build singularity image directly.
**Problem with singularity**
Singularity image is not updated with new packages and seems also impossible to install. 
\
Singularity image: \
normal - cant find dlup \
with no-home - cant find packages


#### Steps with existing singularity image
```bash
sh run_singularity.sh
```
1. Script launches a singularity image file located in container location.
2. It applies all the necessary volume bindings
It can also restrict the container to specified GPUs.


## Slidescore visualisation
For slidescore visualisation of detections (shapes, labels)
Make sure that slidescore-api is installed (should already be installed in setup.)
Then run bash script for upload. An API key that gives access to slidescore studies needs to be generated for each user 
and study. Slidescore also requires the annotations to be uploaded in a certain format.
Install slidescore in separate conda env.
```bash
conda create -n slidescore_env python=3.10
conda activate slidescore_env
git clone slidescore-api
python -m pip install -e slidscore-api
cd DROP/scripts/slidescore_upload_detections
sh upload_precision_to_slidescore.sh
 ```
### For dev
```bash
cd DROP/
black .
pylint --recursive=y
 ```
### Path Setup
To use the repo, the data needs to be provided in the following structure.

Raw images/metadata:
   ```bash
--> orig_data_dir --> dataset_name -->subdir --> images
                                                - img.ext ...
                                             --> metadata 
                                                - slidescore_mapping_fn 
                                                - original metadata file 
--> images in orig_data_dir are rsynced to data_dir for faster access
                                           
 ```
Processed data:
   ```bash
--> derived_data_dir --> dataset_name -->subdir --> h5_embeddings
                                                - embeddings.h5 ...
                                                --> h5_images
                                                - images.h5 ...
                                                --> cached_datasets
                                                - dataset.pkl ...
                                                --> tiff_background_masks
                                                - mask.tiff ...
--> images in orig_data_dir are copied to data_dir for faster access
                                           
 ```
Project data:
   ```bash
--> project_data_dir --> dataset 
                        - matched_metadata_fn
                        - server_id_slide_mapping_fn
                        - metadata_fn (metadata file prepared for project)
                        --> subdir 
                            --> masks 
                                -mask.tiff ...
                        --> bbox_predictions (potentially rename detection)
                            --> img
                                --> subdir
                                    - regions.json (boxes dlup format)
                                    - boxes*.csv (post-processed out file, slidescore)
                                    - tile_bounds.csv
                        --> task
                            - task_data_fn
                            - cv_split_fn
 ```
General project files/ output: 
   ```bash

project_dir --> containers --> singularity image
            --> logs_task --> experiment_name --> split-i --> checkpoints
                                                          --> lighting_logs (for tensorboard)
            --> ssl_d2_ckpt_dir --> ssl_method
                                    - ssl_model_ep*.torch (torch format for cls_)
                                    - ssl_model_ep*_d2.torch (detectron format for det)
 ```
Detection data is currently in /project_dir/detection_data.

### Creation of metadata files

#### Precision_NKI_89_05
1. Save xlsx metadata files as csv - using only the main sheet.
2. make_precision_nki_meta_data_file (in data_proc/meta_data/make_project_data_csv.py)
3. Create meta_data_csv with split using create_split (in data_analysis/meta_data_check/make_project_data_csv.py)
To do this you can run the python file in drop/data_analysis/meta_data_check/make_project_data_csv.py
   
### Models CLS

Models need to be provided to the classification part of the pipeline in torchvision format. 
They can be converted with scripts in model_weight_conversion.


### HYDRA

All parameters will be merged with parameters from default configurations set 
this allows one to overwrite only specified parameters.


### RHPC hyperparameter search
    ```bash

    sbatch scripts/new_cls_hyp/hyperparameter_search.sh
    ```
To view results of Mlflow runs on RHPC, you can use SSH port forwarding to access the MLflow UI in your web browser.
Here are the steps to set up SSH port forwarding:
```bash
ssh rhpc-node -L local_port:localhost:server_port
. start_mlflow.sh
Enter port: local_port
```
Then go to localhost:local_port in browser.
If server_port not available, use another port. 
Find available ports with 

```bash
netstat -atun
```

### LISA hyperparameter search --> Now Snellius

```bash
sbatch hyperparameter_search_LISA.sh
```
Outputs are in the place from which the script is run (general errors and logs - in hyperparameter_master..out/err)
The results of the multirun are stored in the experiment folder/multiruns/.. then the version. In the .submitit folder, 
are the job speciiic logs and errors.

To view MLflow outputs of a currently running run on a remote server, you can use
SSH port forwarding to access the MLflow UI in your web browser. Here are the steps to set up SSH port forwarding:
```bash
ssh -v -N -L local_port:nodename:server_port username@lisa-gpu.surfsara.nl
ssh -v -N -L 5002:r28n4:5002 sdoyle@lisa-gpu.surfsara.nl

```
#### Mlflow results of all runs
Connect to an active node on LISA.
And bind its port to a local port. Then we can view the outputs on the local port.
It will connect to the local_port using some random port on the server_node. 

```bash
 ssh -J lisa sdoyle@server_node -L local_port:localhost:server_port
#on the server_node
source ~/miniconda3/etc/profile.d/conda.sh # might not be necessary
conda activate drop_env  #any env that has mlflow installed
MLFLOW_DIR=/home/sdoyle/to_mlflow_outputs/
mlflow server --backend-store-uri $MLFLOW_DIR --default-artifact-root /home/sdoyle/mlflow_artifacts --host 0.0.0.0:local_port
```
#### Tensorboard on LISA
Same as with viewing mlflow outputs. So we connect to an active node on LISA.
And bind its port to a local port. Then we can view the tensorboard on the local port.

```bash
 ssh -J lisa sdoyle@server_node -L local_port:localhost:server_port
#on the server_node
tensorboard --logdir /path/to/logs --port 6006

```


# for loading dlup with c compiler

source /sw/spack/share/spack/setup-env.sh
spack load pixman@0.40.0 && spack load libvips@8.15.0 &&  spack load openslide@4.0.0  && spack unload py-pip py-wheel py-setuptools python

pip install meson, cython, numpy, pybind11, spin
conda install boost

meson.build:7:0: ERROR: python.find_installation got unknown keyword arguments "pure"


spack unload meson 
python -m pip install meson 

spin build in dlup folder
then python -m pip install . -e

then ninja path error:
~/miniconda3/envs/drop_env_yml/lib/python3.10/site-packages/_dlup_editable_loader.py
replace tmp/ ninja path

/sw/spack/opt/spack/linux-ubuntu20.04-x86_64/gcc-9.4.0/ninja-1.11.1-7cbe4rmakcifit2ttc6zcjax5nqsyevc/bin/ninja

