# -*- coding: utf-8 -*-


def adjust_lr_type1(epoch: int, lr: float) -> float:
    """
    第一种调整学习率\n
    :param epoch: 轮数
    :param lr: 学习率
    :return: 调整后的学习率
    """
    return lr * (0.50 ** (epoch - 1))


def adjust_lr_type2(epoch: int, lr: float) -> float:
    """
    第二种调整学习率\n
    :param epoch: 轮数
    :param lr: 学习率
    :return: 调整后的学习率
    """
    if epoch < 2:
        return lr
    elif epoch < 4:
        return 5e-5
    elif epoch < 6:
        return 1e-5
    elif epoch < 8:
        return 5e-6
    elif epoch < 10:
        return 1e-6
    elif epoch < 15:
        return 5e-7
    elif epoch < 20:
        return 1e-7
    else:
        return 5e-8
