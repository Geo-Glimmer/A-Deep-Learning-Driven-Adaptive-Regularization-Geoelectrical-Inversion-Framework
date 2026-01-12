import tensorflow as tf
from tensorflow.keras import layers, Model, Input, regularizers
import numpy as np

def fcn_denoiser2(
    input_shape=(None, None, 1),
    base_channels=32,
    out_channels=1,
    l2_reg=1e-4,
):
    inputs = Input(shape=input_shape)
    reg = regularizers.l2(l2_reg) if l2_reg > 0 else None
    # Stem（茎）层在深度学习网络中指的是模型的最初几层（通常是第一层或前几层的卷积），
    # 主要作用是接收原始输入并产生特征映射，为后续的深层网络“提取基础特征”做准备。
    # 在本模型中，stem 由一个 Conv2D+ReLU 组成，其输出用于后续残差和ASPP模块。

    # Stem with residual
    stem = layers.Conv2D(base_channels, 3, padding='same', use_bias=False, kernel_regularizer=reg)(inputs)
    stem = layers.ReLU()(stem)
    
    residual = layers.Conv2D(base_channels, 3, padding='same', use_bias=False, kernel_regularizer=reg)(stem)
    residual = layers.ReLU()(residual)

    x = layers.Add()([stem, residual])

    # ASPP with BN
    aspp_branches = []
    b1 = layers.Conv2D(base_channels * 2, 1, padding='same', use_bias=False, kernel_regularizer=reg)(x)
    b1 = layers.ReLU()(b1)
    aspp_branches.append(b1)

    for rate in [2, 3, 4, 5]:
        b = layers.Conv2D(base_channels * 2, 3, padding='same', dilation_rate=rate,
                          use_bias=False, kernel_regularizer=reg)(x)
        b = layers.ReLU()(b)
        b = layers.Dropout(0.01)(b)
        aspp_branches.append(b)

    x = layers.Concatenate(axis=-1)(aspp_branches)
     
 
    # Decoder
    x = layers.Conv2D(base_channels * 2, 3, padding='same', use_bias=False, kernel_regularizer=reg)(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(1e-2)(x)

    x = layers.Conv2D(base_channels, 3, padding='same', use_bias=False, kernel_regularizer=reg)(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(1e-2)(x)

    main_output = layers.Conv2D(out_channels, 1, name='main_output')(x)

    return Model(inputs=inputs, outputs=main_output, name='fcn_denoiser_v2_reg')