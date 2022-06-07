# -*-Encoding: utf-8 -*-
from tensorflow.keras.callbacks import Callback


class OneTurbCallback(Callback):
    """
    完成或开始一个风机的回调\n
    """

    def __init__(self, turb_id: int, train_begin: bool = False, train_end: bool = False, predict_begin: bool = False, predict_end: bool = False):
        super().__init__()
        self.turb_id = turb_id
        self.train_begin = train_begin
        self.train_end = train_end
        self.predict_begin = predict_begin
        self.predict_end = predict_end

    def on_train_begin(self, logs=None):
        if self.train_begin:
            print(">>>>>>>>>>>>> turb %d training begin >>>>>>>>>>>>>>>>>" % self.turb_id)
        return super().on_train_begin(logs)

    def on_train_end(self, logs=None):
        if self.train_end:
            print(">>>>>>>>>>>>> turb %d training end >>>>>>>>>>>>>>>>>" % self.turb_id)
        return super().on_train_end(logs)

    def on_predict_begin(self, logs=None):
        if self.predict_begin:
            print(">>>>>>>>>>>>> turb %d predicting begin >>>>>>>>>>>>>>>>>" % self.turb_id)
        return super().on_predict_begin(logs)

    def on_predict_end(self, logs=None):
        if self.predict_end:
            print(">>>>>>>>>>>>> turb %d predicting end >>>>>>>>>>>>>>>>>" % self.turb_id)
        return super().on_predict_end(logs)
