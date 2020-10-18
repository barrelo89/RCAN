import cv2
import numpy as np
import tensorflow as tf

class ConvBlock(tf.keras.layers.Layer):

    def __init__(self, kernel_size):
        super(ConvBlock, self).__init__()
        self.kernel_size = kernel_size

    def build(self, input_shape):

        _, _, _, num_channel = input_shape
        self.conv_1 = tf.keras.layers.Conv2D(filters = num_channel, kernel_size = self.kernel_size, padding = 'same')
        self.conv_2 = tf.keras.layers.Conv2D(filters = num_channel, kernel_size = self.kernel_size, padding = 'same')
        self.relu = tf.keras.layers.ReLU()

    def call(self, input):
        x = self.conv_1(input)
        x = self.relu(x)
        x = self.conv_2(x)

        return x

class CA(tf.keras.layers.Layer): #Channel Attention

    def __init__(self, kernel_size, reduction_rate):
        super(CA, self).__init__()
        self.kernel_size = kernel_size
        self.global_pooling = tf.keras.layers.GlobalAveragePooling2D()
        self.reduction_rate = reduction_rate

    def build(self, input_shape):

        _, _, _, num_channel = input_shape
        self.conv_in = tf.keras.layers.Conv2D(filters = int(num_channel / reduction_rate), kernel_size = self.kernel_size, padding = 'same')
        self.relu = tf.keras.layers.ReLU()
        self.conv_out = tf.keras.layers.Conv2D(filters = num_channel, kernel_size = self.kernel_size, padding = 'same')

    def call(self, input):

        x = self.global_pooling(input)
        x = tf.reshape(x, (-1, 1, 1, x.shape[-1]))
        x = self.conv_in(x)
        x = self.relu(x)
        x = self.conv_out(x)
        x = tf.keras.activations.sigmoid(x)

        return tf.math.multiply(input, x)

class RCAB(tf.keras.layers.Layer):

    def __init__(self, kernel_size, scale_factor, reduction_rate):
        super(RCAB, self).__init__()
        self.scale_factor = scale_factor
        self.conv_block = ConvBlock(kernel_size)
        self.channel_attention = CA(kernel_size, reduction_rate)

    def call(self, input):
        x = self.conv_block(input)
        return input + self.channel_attention(x)*self.scale_factor

class ResGroup(tf.keras.layers.Layer):

    def __init__(self, num_block, kernel_size, scale_factor, reduction_rate):
        super(ResGroup, self).__init__()

        self.scale_factor = scale_factor
        self.kernel_size = kernel_size
        self.num_block = num_block
        self.reduction_rate = reduction_rate

    def build(self, input_shape):

        _, _, _, num_channel = input_shape

        self.block_list = []
        for _ in range(num_block):
            self.block_list.append(RCAB(self.kernel_size, self.scale_factor, self.reduction_rate))
        self.conv_out = tf.keras.layers.Conv2D(filters = num_channel, kernel_size = self.kernel_size, padding = 'same')

    def call(self, input):

        x = input

        for rcab in self.block_list:
            x = rcab(x)
        x = self.conv_out(x)

        return input + x*self.scale_factor

class Upscale(tf.keras.layers.Layer):

    def __init__(self, kernel_size, up_scale_factor, squeeze_factor):
        super(Upscale, self).__init__()
        self.kernel_size = kernel_size
        self.up_scale_factor = up_scale_factor
        self.squeeze_factor = squeeze_factor

    def build(self, input_shape):

        _, _, _, num_channel = input_shape
        self.convt = tf.keras.layers.Conv2DTranspose(filters = int(num_channel / self.squeeze_factor), kernel_size = self.kernel_size, strides = self.up_scale_factor, padding = 'same')

    def call(self, input):
        return self.convt(input)

class RCAN(tf.keras.Model):

    def __init__(self, initial_num_filters, num_group, num_block, kernel_size, scale_factor, up_scale_factor, reduction_rate, squeeze_factor):
        super(RCAN, self).__init__()
        self.in_conv = tf.keras.layers.Conv2D(filters = initial_num_filters, kernel_size = kernel_size, padding = 'same')

        self.group_list = []
        for _ in range(num_group):
            self.group_list.append(ResGroup(num_block, kernel_size, scale_factor, reduction_rate))
        self.mid_conv = tf.keras.layers.Conv2D(filters = initial_num_filters, kernel_size = kernel_size, padding = 'same')
        self.up_scale = Upscale(kernel_size, up_scale_factor, squeeze_factor)
        self.out_conv = tf.keras.layers.Conv2D(filters = 3, kernel_size = kernel_size, padding = 'same')

    def call(self, input):

        conv_x = self.in_conv(input)
        x = conv_x

        for res_group in self.group_list:
            x = res_group(x)

        x = self.mid_conv(x)
        x += conv_x

        x = self.up_scale(x)
        x = self.out_conv(x)

        x = tf.clip_by_value(x, 0.0, 255.0)
        return x

'''
MINIMIZE L1 LOSS
'''

initial_num_filters = 64
num_group = 3
num_block = 3
kernel_size = 3
scale_factor = 0.1
up_scale_factor = 2
reduction_rate = 3
squeeze_factor = 2

data = cv2.imread('img.jpg').astype(float)
small_data = cv2.resize(data, None, fx = 1/up_scale_factor, fy = 1/up_scale_factor)

model = RCAN(initial_num_filters, num_group, num_block, kernel_size, scale_factor, up_scale_factor, reduction_rate, squeeze_factor)

optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4)
model.compile(optimizer = optimizer, loss = tf.keras.losses.MeanAbsoluteError())

model.fit(small_data[np.newaxis, ...], data[np.newaxis, ...], epochs = 1000)

result = model.predict(small_data[np.newaxis, ...])
result = tf.cast(result, tf.uint8)

cv2.imshow('img', result.numpy()[0, :, :, :])
cv2.waitKey(0)








































#end
