import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def _welford_update(existing_aggregate, X_y_dataset):

    X_batched_dataset = X_y_dataset[:][0]

    (count, mean, M2) = existing_aggregate
    if count is None:
        count, mean, M2 = 0, np.zeros_like(X_batched_dataset[0]), np.zeros_like(X_batched_dataset[0])

    for i in range(X_batched_dataset.shape[0]):
        img = X_batched_dataset[i][:][:]
        count += 1
        delta = img - mean
        mean += delta / count
        delta2 = img - mean
        M2 += delta * delta2

    return (count, mean, M2)

def _welford_finalize(existing_aggregate):
    count, mean, M2 = existing_aggregate
    mean, variance, sample_variance = (mean, M2/count, M2/(count - 1))
    if count < 2:
        return (float("nan"), float("nan"), float("nan"))
    else:
        return (mean, variance, sample_variance)


def welford(sample):
    """Calculates the mean, variance and sample variance along the first axis of an array.
    Taken from https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    """
    existing_aggregate = (None, None, None)

    for i in range(sample.__len__()):
        existing_aggregate = _welford_update(existing_aggregate, sample.__getitem__(i))

    # sample variance only for calculation
    return _welford_finalize(existing_aggregate)[:-1]

def dct2(array, batched=True):
    """DCT-2D transform of an array, by first applying dct-2 along rows then columns
    Arguments:
        array - the array to transform.
    Returns:
        DCT2D transformed array.
    """
    shape = array.shape
    dtype = array.dtype
    array = tf.cast(array, tf.float32)

    if batched:
        # tensorflow computes over last axis (-1)
        # layout (B)atch, (R)ows, (C)olumns, (V)alue
        # BRCV
        array = tf.transpose(array, perm=[0, 3, 2, 1])
        # BVCR
        array = tf.signal.dct(array, type=2, norm="ortho")
        array = tf.transpose(array, perm=[0, 1, 3, 2])
        # BVRC
        array = tf.signal.dct(array, type=2, norm="ortho")
        array = tf.transpose(array, perm=[0, 2, 3, 1])
        # BRCV
    else:
        # RCV
        array = tf.transpose(array, perm=[2, 1, 0])
        # VCR
        array = tf.signal.dct(array, type=2, norm="ortho")
        array = tf.transpose(array, perm=[0, 2, 1])
        # VRC
        array = tf.signal.dct(array, type=2, norm="ortho")
        array = tf.transpose(array, perm=[1, 2, 0])
        # RCV

    array = tf.cast(array, dtype)

    array.shape.assert_is_compatible_with(shape)

    return array

class DCTLayer(layers.Layer):
    
    def __init__(self, mean, var, **kwargs):
        super(DCTLayer, self).__init__()
        self.mean = mean
        self.var = var
        self.std = np.sqrt(var)

    def build(self, input_shape):
        self.mean_w = self.add_weight('mean', shape=np.shape(self.mean), initializer=keras.initializers.Constant(self.mean), trainable=False)
        self.std_w = self.add_weight('std', shape=np.shape(self.mean), initializer=keras.initializers.Constant(self.std), trainable=False)

    def call(self, inputs):

        x = dct2(inputs)

        x = tf.abs(x)
        x += 1e-13
        x = tf.math.log(x)

        x = x - self.mean_w
        x = x / self.std_w

        return x

    def get_config(self):

        config = super().get_config()
        
        config.update({
            "mean": self.mean,
            "var": self.var,
        })
        
        return config

def CNN_DCT(mean, var, input_shape=(200, 200, 3), classes=2):

    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(3, 3, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(8, 3, padding='same', activation='relu')(x)
    x = layers.AveragePooling2D()(x)

    x = layers.Conv2D(6, 3, padding='same', activation='relu')(x)
    x = layers.AveragePooling2D()(x)

    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)

    x = layers.Flatten()(x)

    outputs = layers.Dense(classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs)

    model = keras.models.Sequential([DCTLayer(mean, var), model])

    return model
