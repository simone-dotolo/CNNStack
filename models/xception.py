import tensorflow as tf

def Xception(input_shape=(200, 200, 3), classes=2):

    base_model = tf.keras.applications.Xception(include_top=False, weights='imagenet', input_shape=input_shape)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    model.add(tf.keras.layers.Rescaling(1./127.5, offset=-1))
    model.add(base_model)
    model.add(tf.keras.layers.GlobalMaxPooling2D())
    model.add(tf.keras.layers.Dense(classes, activation='softmax'))

    return model

