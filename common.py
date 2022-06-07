# -*-Encoding: utf-8 -*-
from typing import Optional, Union
from pandas import DataFrame, Series
import numpy as np
from numpy import ndarray
import tensorflow as tf


class Scaler(object):
    """
    Desc: Normalization utilities\n
    """

    def __init__(self, _settings: dict):
        self.mean = 0.
        self.std = 1.
        self.target = _settings["target"]

    def fit(self, data: Union[ndarray, DataFrame, Series]) -> None:
        """
        制作一个标准化器\n
        :param data: 输入
        :return
        """
        self.mean = data.mean()
        self.std = data.std()

    def transform(self, data: Union[ndarray, DataFrame, Series]) -> Union[ndarray, DataFrame, Series]:
        """
        Desc:
            Transform the data
        Args:
            data:
        Returns:
            The transformed data
        """
        mean = self.mean
        std = self.std
        return (data - mean) / std

    def inverse_transform(self, data: Union[ndarray, DataFrame, Series], only_target: bool = False) -> Union[ndarray, DataFrame, Series]:
        """
        Desc:
            Restore to the original data
        Args:
            data: the transformed data
        Returns:
            The original data
        """
        mean = self.mean
        std = self.std
        if only_target:
            return (data * std.loc[self.target]) + mean.loc[self.target]
        else:
            return (data * std) + mean


def windowed_dataset(
        series: np.ndarray,
        sequence_length: int,
        x_end_index: int,
        y_start_index: Optional[int],
        x_dim: Optional[int] = None,
        y_dim: Optional[int] = 1,
        batch_size: int = 1,
        shuffle_buffer: Optional[int] = 10
) -> tf.raw_ops.PrefetchDataset:
    """
    滑动窗口划分数据集\n
    :param series: 时间序列
    :param sequence_length: 窗口长度，包括x和y
    :param x_end_index: x结束的索引
    :param y_start_index: y开始的索引。如果只有自变量则该参数为None，此时x_end_index参数无效
    :param x_dim: 输入的维度
    :param y_dim: 输出的维度
    :param batch_size: 批量大小
    :param shuffle_buffer: 洗牌缓冲区大小，设置为1000即可。如果不需要洗牌，将该参数设置为None或0
    :return: 划分后的时间序列
    """
    ds: tf.raw_ops.TensorSliceDataset = tf.data.Dataset.from_tensor_slices(series.astype("float32"))
    if y_start_index is not None:
        ds: tf.raw_ops.WindowDataset = ds.window(sequence_length, shift=1, drop_remainder=True)
        ds: tf.raw_ops.FlatMapDataset = ds.flat_map(lambda w: w.batch(sequence_length))
        if shuffle_buffer:
            ds: tf.raw_ops.ShuffleDataset = ds.shuffle(shuffle_buffer)
        ds: tf.raw_ops.MapDataset = ds.map(lambda w: (w[0:x_end_index, 0:x_dim], w[y_start_index:, -y_dim:]))
    else:
        ds: tf.raw_ops.WindowDataset = ds.window(sequence_length, shift=1, drop_remainder=True)
        ds: tf.raw_ops.FlatMapDataset = ds.flat_map(lambda w: w.batch(sequence_length))
        if shuffle_buffer:
            ds: tf.raw_ops.ShuffleDataset = ds.shuffle(shuffle_buffer)
    ds: tf.raw_ops.BatchDataset = ds.batch(batch_size, drop_remainder=False)
    return ds.prefetch(1)


def get_turb(data: DataFrame, turb_id: int, _settings: dict) -> (tf.raw_ops.PrefetchDataset, ):
    """
    获取一个风机的数据\n
    :param data: 原始数据
    :param turb_id: 风机id
    :param _settings: 设置项
    :return: train_data, val_data
    """
    turbine_id = _settings["turbine_id"]
    start_col = _settings["start_col"]
    input_len = _settings["input_len"]
    output_len = _settings["output_len"]
    in_var = _settings["in_var"]
    out_var = _settings["out_var"]
    train_size = _settings["train_size"]
    data = data[data[turbine_id] == turb_id]
    scl = Scaler(_settings)
    scl.fit(data.iloc[:, start_col:])
    data.iloc[:, start_col:] = scl.transform(data.iloc[:, start_col:])
    xy_train = data[data["Day"] <= train_size].fillna(0).iloc[:, start_col:]
    xy_val = data[data["Day"] > train_size].fillna(0).iloc[:, start_col:]
    return windowed_dataset(
        series=xy_train.values,
        sequence_length=input_len + output_len,
        x_end_index=input_len,
        y_start_index=input_len,
        x_dim=in_var,
        y_dim=out_var,
        batch_size=_settings["batch_size"]
    ), windowed_dataset(
        series=xy_val.values,
        sequence_length=input_len + output_len,
        x_end_index=input_len,
        y_start_index=input_len,
        x_dim=in_var,
        y_dim=out_var,
        batch_size=_settings["batch_size"]
    )


def get_turb_test(data: DataFrame, turb_id: int, _settings: dict) -> (tf.raw_ops.PrefetchDataset, Scaler):
    """
    获取一个风机的数据\n
    :param data: 原始数据
    :param turb_id: 风机id
    :param _settings: 设置项
    :return: 测试集x
    """
    turbine_id = _settings["turbine_id"]
    start_col = _settings["start_col"]
    input_len = _settings["input_len"]
    output_len = _settings["output_len"]
    in_var = _settings["in_var"]
    out_var = _settings["out_var"]
    data = data[data[turbine_id] == turb_id]
    scl = Scaler(_settings)
    scl.fit(data.iloc[:, start_col:])
    data.iloc[:, start_col:] = scl.transform(data.iloc[:, start_col:])
    xy_train = data.fillna(0).iloc[-input_len:, start_col:]
    return windowed_dataset(
        series=xy_train.values,
        sequence_length=input_len,
        x_end_index=input_len,
        y_start_index=None,
        x_dim=in_var,
        y_dim=out_var,
        batch_size=_settings["batch_size"]
    ), scl
