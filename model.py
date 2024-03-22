from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, ReLU, Add, MaxPooling1D, GlobalAveragePooling1D, \
    Dense
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


def residual_block(x, filters, kernel_size=3, stride=1, conv_shortcut=False, name=None):
    """
    A residual block.

    Arguments:
    - x: input tensor.
    - filters: integer, filters of the convolution.
    - kernel_size: size of the kernel for the convolution.
    - stride: stride of the convolution.
    - conv_shortcut: Boolean, if True, adds a convolutional layer for shortcut connection.
    - name: name of the block.

    Returns:
    - Output tensor for the block.
    """
    if conv_shortcut:
        shortcut = Conv1D(filters, 1, strides=stride, kernel_regularizer=l2(1e-4), name=name + '_shortcut')(x)
        shortcut = BatchNormalization(name=name + '_shortcut_bn')(shortcut)
    else:
        shortcut = x

    # First convolution
    x = Conv1D(filters, kernel_size, strides=stride, padding='same', kernel_regularizer=l2(1e-4), name=name + '_conv1')(
        x)
    x = BatchNormalization(name=name + '_conv1_bn')(x)
    x = ReLU(name=name + '_conv1_relu')(x)

    # Second convolution
    x = Conv1D(filters, kernel_size, padding='same', kernel_regularizer=l2(1e-4), name=name + '_conv2')(x)
    x = BatchNormalization(name=name + '_conv2_bn')(x)

    # Add shortcut
    x = Add(name=name + '_add')([shortcut, x])
    x = ReLU(name=name + '_out_relu')(x)

    return x


def build_resnet(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    x = Conv1D(64, 7, strides=2, padding='same', kernel_regularizer=l2(1e-4), name='conv1')(inputs)
    x = BatchNormalization(name='conv1_bn')(x)
    x = ReLU(name='conv1_relu')(x)
    x = MaxPooling1D(3, strides=2, padding='same', name='pool1')(x)

    # Residual blocks
    x = residual_block(x, 64, conv_shortcut=True, name='block1')
    x = residual_block(x, 128, stride=2, conv_shortcut=True, name='block2')
    x = residual_block(x, 256, stride=2, conv_shortcut=True, name='block3')
    x = residual_block(x, 512, stride=2, conv_shortcut=True, name='block4')

    x = GlobalAveragePooling1D(name='avg_pool')(x)
    outputs = Dense(num_classes, activation='softmax', name='fc')(x)

    model = Model(inputs, outputs, name='resnet1d')
    return model
