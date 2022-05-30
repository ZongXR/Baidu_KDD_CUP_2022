# -*-Encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Description: A demo of the forecasting method
Authors: Lu,Xinjiang (luxinjiang@baidu.com)
Date:    2022/04/18
"""
import os
import time
import numpy as np
import paddle
from model import BaselineGruModel
from common import Experiment, traverse_wind_farm
from wind_turbine_data import WindTurbineData
from test_data import TestData


def forecast_one(experiment, test_turbines, train_data):
    # type: (Experiment, TestData, WindTurbineData) -> np.ndarray
    """
    Desc:
        Forecasting the power of one turbine
    Args:
        experiment:
        test_turbines:
        train_data:
    Returns:
        Prediction for one turbine
    """
    args = experiment.get_args()
    tid = args["turbine_id"]
    model = BaselineGruModel(args)
    model_dir = '{}_t{}_i{}_o{}_ls{}_train{}_val{}'.format(
        args["filename"], args["task"], args["input_len"], args["output_len"], args["lstm_layer"],
        args["train_size"], args["val_size"]
    )
    path_to_model = os.path.join(args["checkpoints"], model_dir, "model_{}".format(str(tid)))
    model.set_state_dict(paddle.load(path_to_model))

    test_x, _ = test_turbines.get_turbine(tid)
    scaler = train_data.get_scaler(tid)
    test_x = scaler.transform(test_x)

    last_observ = test_x[-args["input_len"]:]
    seq_x = paddle.to_tensor(last_observ)
    sample_x = paddle.reshape(seq_x, [-1, seq_x.shape[-2], seq_x.shape[-1]])
    prediction = experiment.inference_one_sample(model, sample_x)
    prediction = scaler.inverse_transform(prediction)
    prediction = prediction[0]
    return prediction.numpy()


def forecast(settings):
    # type: (dict) -> np.ndarray
    """
    Desc:
        Forecasting the wind power in a naive distributed manner
    Args:
        settings:
    Returns:
        The predictions as a tensor \in R^{134 * 288 * 1}
    """
    start_time = time.time()
    predictions = []
    settings["turbine_id"] = 0
    exp = Experiment(settings)
    # train_data = Experiment.train_data
    train_data = exp.load_train_data()
    if settings["is_debug"]:
        end_train_data_get_time = time.time()
        print("Load train data in {} secs".format(end_train_data_get_time - start_time))
        start_time = end_train_data_get_time
    test_x = Experiment.get_test_x(settings)
    if settings["is_debug"]:
        end_test_x_get_time = time.time()
        print("Get test x in {} secs".format(end_test_x_get_time - start_time))
        start_time = end_test_x_get_time
    for i in range(settings["capacity"]):
        settings["turbine_id"] = i
        # print('\n>>>>>>> Testing Turbine {:3d} >>>>>>>>>>>>>>>>>>>>>>>>>>\n'.format(i))
        prediction = forecast_one(exp, test_x, train_data)
        paddle.device.cuda.empty_cache()
        predictions.append(prediction)
        if settings["is_debug"] and (i + 1) % 10 == 0:
            end_time = time.time()
            print("\nElapsed time for predicting 10 turbines is {} secs".format(end_time - start_time))
            start_time = end_time
    return np.array(predictions)


def validate_one(experiment, model_folder):
    # type: (Experiment, str) -> (paddle.tensor, paddle.tensor)
    """
    Desc:
        Forecasting the power for one turbine
    Args:
        experiment:
        model_folder: the location of the model
    Returns:
        MAE and RMSE
    """
    args = experiment.get_args()
    model = experiment.get_model()
    path_to_model = os.path.join(args["checkpoints"], model_folder, 'model_{}'.format(str(args["turbine_id"])))
    model.set_state_dict(paddle.load(path_to_model))

    test_data, test_loader = experiment.get_data(flag='test')
    predictions = []
    true_lst = []
    for i, (batch_x, batch_y) in enumerate(test_loader):
        sample, true = experiment.process_one_batch(batch_x, batch_y)
        predictions.append(np.array(sample))
        true_lst.append(np.array(true))
    predictions = np.array(predictions)
    true_lst = np.array(true_lst)
    predictions = predictions.reshape(-1, predictions.shape[-2], predictions.shape[-1])
    true_lst = true_lst.reshape(-1, true_lst.shape[-2], true_lst.shape[-1])

    predictions = test_data.inverse_transform(predictions)
    true_lst = test_data.inverse_transform(true_lst)
    raw_df = test_data.get_raw_data()

    return predictions, true_lst, raw_df


def validate(settings):
    # type: (dict) -> tuple
    """
    Desc:
        Forecasting the wind power in a naive distributed manner
    Args:
        settings:
    Returns:
        The predictions and the ground truths
    """
    preds = []
    gts = []
    raw_data_ls = []
    cur_setup = '{}_t{}_i{}_o{}_ls{}_train{}_val{}'.format(
        settings["filename"], settings["task"], settings["input_len"], settings["output_len"], settings["lstm_layer"],
        settings["train_size"], settings["val_size"]
    )
    results = traverse_wind_farm(validate_one, settings, cur_setup, flag='test')
    for j in range(settings["capacity"]):
        pred, gt, raw_data = results[j]
        preds.append(pred)
        gts.append(gt)
        raw_data_ls.append(raw_data)

    return preds, gts, raw_data_ls
