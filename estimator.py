# -*-Encoding: utf-8 -*-
import os
from time import strftime, localtime
from copy import deepcopy
from typing import List, Union, Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, BatchNormalization, TimeDistributed, Dropout, LSTM, GRU, Lambda, MaxPool1D, Bidirectional, Conv1D, AvgPool1D
from tensorflow.keras.optimizers import Optimizer, Adam
from tensorflow.keras.callbacks import History, TensorBoard, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.metrics import Metric
from tensorflow.keras.losses import Loss, Huber, MeanSquaredError, Reduction
from tensorflow.keras.backend import clear_session
from tensorflow.keras.utils import serialize_keras_object
from callbacks import OneTurbCallback
from lr_adjust import adjust_lr_type1, adjust_lr_type2


class WPF(Model):
    """
    LSTM模型\n
    """

    # 清除计算图
    tf.compat.v1.reset_default_graph()
    clear_session()

    def __init__(self, turb_id: int, settings: dict) -> None:
        """
        构造方法\n
        :param turb_id: 风机id
        :param settings: 配置
        """
        super(WPF, self).__init__(name="LSTM")
        self.turb_id = turb_id
        self.input_len = settings["input_len"]
        self.output_len = settings["output_len"]
        self.train_epochs = settings["train_epochs"]
        self.num_workers = settings["num_workers"]
        self.logdir = settings["logdir"]
        self.in_var = settings["in_var"]
        self.out_var = settings["out_var"]
        self.checkpoints = settings["checkpoints"]
        self.patience = settings["patience"]
        self.lr = settings["lr"]
        self.is_debug = settings["is_debug"]
        self.dropout = settings["dropout"]
        self.lr_adjust = settings["lr_adjust"]

        self.pad_1 = Lambda(function=lambda x: tf.pad(x, [[0, 0], [0, self.output_len], [0, 0]], mode="CONSTANT", constant_values=0), name="pad_1")
        self.gru_1 = GRU(units=64, return_sequences=True, return_state=False, name="gru_1")
        self.gru_2 = GRU(units=64, return_sequences=True, return_state=False, name="gru_2")
        self.dropout_3 = Dropout(rate=self.dropout, name="dropout_3")
        self.linear_4 = Dense(units=self.out_var, activation=None, name="linear_4")
        self.output_layer_5 = Lambda(function=lambda x: x[:, -self.output_len:, :], name="output_5")

    def get_config(self):
        layer_configs = []
        for layer in self.layers:
            layer_configs.append(serialize_keras_object(layer))
        config = {
            'name': self.name,
            'layers': deepcopy(layer_configs)
        }
        if not self._is_graph_network and self._build_input_shape is not None:
            config['build_input_shape'] = self._build_input_shape
        return config

    def call(self, inputs, training=None, mask=None):
        result = self.pad_1(inputs)
        result = self.gru_1(result)
        result = self.gru_2(result)
        result = self.dropout_3(result)
        result = self.linear_4(result)
        result = self.output_layer_5(result)
        return result

    def load_weights(
            self,
            filename,
            by_name=False,
            skip_mismatch=False,
            options=None
    ):
        """
        加载权重\n
        :param filename: 文件名，不包括路径
        :param by_name: 是否按层名称加载
        :param skip_mismatch: 是否跳过不匹配的
        :param options: 选项
        :return:
        """
        self(np.zeros((1, self.input_len, self.in_var)))
        return super().load_weights(
            filepath=os.path.join(self.checkpoints, filename),
            by_name=by_name,
            skip_mismatch=skip_mismatch,
            options=options
        )

    def compile(
            self,
            optimizer: Union[str, Optimizer] = "Adam",
            loss: Union[str, Loss] = "Huber",
            metrics: Union[Tuple[str], Tuple[Metric]] = ("mse", ),
            **kwargs
    ):
        """
        编译模型\n
        :param optimizer: 优化器
        :param loss: 损失函数
        :param metrics: 度量
        :param kwargs: 其他参数
        :return:
        """
        result = super().compile(
            optimizer=Adam(learning_rate=self.lr, clipnorm=50.0),
            loss=MeanSquaredError(reduction=Reduction.SUM_OVER_BATCH_SIZE),
            metrics=metrics,
            **kwargs
        )
        # self.build(input_shape=(None, None, self.in_var))
        self(np.zeros((1, self.input_len, self.in_var)))
        return result

    def fit(self, x: tf.raw_ops.PrefetchDataset = None, validation_data=None, **kwargs) -> History:
        """
        训练模型\n
        :param x: 数据
        :param validation_data: 验证集
        :param kwargs: 其它参数
        :return: 训练过程
        """
        return super().fit(
            x=x,
            validation_data=validation_data,
            verbose=int(self.is_debug),
            epochs=self.train_epochs,
            workers=self.num_workers,
            use_multiprocessing=True,
            callbacks=[
                TensorBoard(log_dir=os.path.join(self.logdir, strftime("%Y%m%d_%H.%M.%S", localtime())), write_graph=True, write_images=True),
                EarlyStopping(monitor="val_loss", patience=self.patience, restore_best_weights=True),
                # LearningRateScheduler(schedule=adjust_lr_type1 if self.lr_adjust == "type1" else adjust_lr_type2, verbose=int(self.is_debug)),
                ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=int(self.is_debug)),
                OneTurbCallback(self.turb_id, train_begin=True)
            ],
            **kwargs
        )

    def save_weights(self, filename: str, overwrite=True, **kwargs):
        """
        保存模型\n
        :param filename: 文件名，不包含路径
        :param overwrite: 是否覆盖
        :param kwargs: 其它参数
        :return:
        """
        return super().save_weights(os.path.join(self.checkpoints, filename), overwrite, **kwargs)

    def predict(self, x: tf.raw_ops.PrefetchDataset = None, **kwargs) -> np.ndarray:
        """
        预测\n
        :param x: x_test
        :param kwargs: 其它参数
        :return: 预测的结果
        """
        return super().predict(
            x=x,
            verbose=int(self.is_debug),
            workers=self.num_workers,
            use_multiprocessing=True,
            callbacks=[
                OneTurbCallback(self.turb_id, predict_begin=True)
            ]
        )



