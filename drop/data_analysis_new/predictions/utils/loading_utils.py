import pandas as pd
import logging
import omegaconf
import mlflow
from drop.tools.json_saver import JsonSaver


def get_meta_df(meta_path, meta_fn="avail_slides_ssid_mask_metadata_matched.csv"):
    meta_df = pd.read_csv(
        f"{meta_path}{meta_fn}", dtype={"imageName": str, "imageID": str}
    ).reset_index(drop=True)
    return meta_df

def get_predctions_json(run_path, fold=None):
    predictions_fn = "predictions.json"
    if type(fold) == int:
        predictions_path = f"{run_path}metrics/fold{fold}/{predictions_fn}"
    else:
        predictions_path = f"{run_path}metrics/{predictions_fn}"
    json_saver = JsonSaver("predictions", predictions_path)
    return json_saver

def get_cfg_run(model_path):
    config_path = f"{model_path}/.hydra/config.yaml"
    cfg_run = omegaconf.OmegaConf.load(config_path)
    return cfg_run


def read_slide_predictions(run_path, epoch=0, fold=None, stage=None, subdirs=None, ensemble=False,
                           train_without_val=False):
    """"
    Read predictions from json file.
    For debug use : res = json_saver.read_json()
    If the model is trained on an ensemble model, or whole train data --> epoch should be 0.
    """
    slide_level_results ={}
    json_saver = get_predctions_json(run_path, fold=fold)
    id_dict = {"ensemble": ensemble, "train_without_val": train_without_val}
    if stage is not None:
        id_dict["stage"] = stage
    if subdirs is not None:
        id_dict["subdirs"] = subdirs
    try:
        id_dict['epoch'] = int(epoch)
        slide_level_results = json_saver.read_selected_data(id_dict)[0]["slide_level_results"]
    except:
        res = json_saver.read_json()
        # using reversed, because the test entry should actually be the last entry
        for i in reversed(res):
            id_dict_res =  i['predictions']
            dropped_epoch =  id_dict_res.pop('epoch')
            if id_dict_res == id_dict:
                id_dict['epoch'] = dropped_epoch
                slide_level_results = json_saver.read_selected_data(id_dict)[0]["slide_level_results"]
                break

    slide_df = pd.DataFrame(slide_level_results)
    slide_df["imageName"] = slide_df.index
    return slide_df

