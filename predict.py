# -*-Encoding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from estimator import WPF
from common import get_turb_test
from prepare import prep_env


physical_devices = tf.config.list_physical_devices('GPU')
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, enable=True)


def forecast(_settings: dict) -> np.ndarray:
    """
    预测结果\n
    :param _settings: 配置
    :return: 预测值
    """
    capacity = _settings["capacity"]
    output_len = _settings["output_len"]
    out_var = _settings["out_var"]
    results = np.zeros(shape=(capacity, output_len, out_var), dtype=float)
    turbine_id = _settings["turbine_id"]
    filepath = os.path.join(_settings["path_to_test_x"], _settings["file_to_test_x"])
    data = pd.read_csv(filepath)
    turb_ids = data[turbine_id].unique().tolist()
    model = WPF(0, _settings)
    for i, turb_id in enumerate(turb_ids):
        model.turb_id = turb_id
        model.load_weights("model_%d.h5" % turb_id)
        data_test, scaler = get_turb_test(data, turb_id, _settings)
        result = model.predict(data_test)
        result = scaler.inverse_transform(result, only_target=True)
        results[i, :, 0] = result.ravel()
    return np.array(results)


if __name__ == '__main__':
    settings = prep_env()

    print(forecast(settings).shape)
