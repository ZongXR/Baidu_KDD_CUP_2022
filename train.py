# -*-Encoding: utf-8 -*-
import os
import pandas as pd
import tensorflow as tf
from prepare import prep_env
from common import get_turb
from estimator import WPF


physical_devices = tf.config.list_physical_devices('GPU')
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, enable=True)


def train_and_val(_settings: dict) -> None:
    """
    训练模型\n
    :param _settings: 关键字参数
    :return: 空
    """
    turbine_id = _settings["turbine_id"]
    filepath = os.path.join(_settings["data_path"], _settings["filename"])
    data = pd.read_csv(filepath)
    turb_ids = data[turbine_id].unique().tolist()
    for turb_id in turb_ids:
        data_train, data_val = get_turb(data, turb_id, _settings)
        model = WPF(turb_id, _settings)
        model.compile()
        model.summary()
        model.fit(data_train, data_val)
        model.save_weights("model_%d.h5" % turb_id)


if __name__ == "__main__":
    settings = prep_env()

    train_and_val(settings)

