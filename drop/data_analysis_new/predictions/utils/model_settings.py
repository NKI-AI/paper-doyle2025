
def get_model_details(model, outer_cross_val_fold = None, years=False):
    """
    multi_run indicates whether it was a multirun and also which multirun it was.
    We do not use years at the moment, so clinical models are automatically years=false
    """
    model_name, survival, multirun = None, None, None
    if model == "outer_cross_val_int":
        model_name = f'Folds_correct_outer_cross{outer_cross_val_fold}_val_intBlock1_Aperio_ExtraSlides_512_200'
        survival = {"model_type": "dl"}
    elif model == "outer_cross_val_img":
        model_name = f'Folds_correct_outer_cross{outer_cross_val_fold}_val_imgBlock1_Aperio_ExtraSlides_512_200'
        survival = {"model_type": "dl"}
    elif model == "clinical_basic":
        model_name = (f"Clinical_only_outersplit{outer_cross_val_fold}COXPH_months_acc_noNAs_withouther2borderBlock1_Aperio_ExtraSlides_basic_years{years}_final")
        survival = {"model_type": "COXPH"}
    elif model == "clinical_extended":
        model_name = (f"Clinical_ext_only_outersplit_bin{outer_cross_val_fold}COXPH_months_acc_noNAs_withouther2borderBlock1_Aperio_ExtraSlides_extended_years{years}_final")
        survival = {"model_type": "COXPH"}
    else:
        raise ValueError(f"Model {model} not recognised")
    if "clinical" in model:
        survival = {"model_type": "COXPH"}
        multirun = False
    return model_name, survival, multirun
