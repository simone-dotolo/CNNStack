import argparse
import scipy.ndimage as ndi
import numpy as np
import tensorflow as tf
from models.resnet import ResNet50
from models.cnndct import CNN_DCT
from models.cnndct import welford
from models.xception import Xception
from models.efficientnetb4 import EfficientNetB4


# Preprocessing
sigma_min = 0
sigma_max = 3
rotation_range = 90

# Training
beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-07
patience = 15
patience_lr = 10
factor = 0.1

input_shape = (200,200,3)
classes = 2

def preprocessing_function(x):
    #x = ndi.gaussian_filter(x, sigma=np.random.randint(sigma_min, sigma_max + 1), truncate=1)
    #x = x/127.5 - 1
    return x

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Model to train', required=True)
    parser.add_argument('--train_fold', type=str, help='Path to training set', required=True)
    parser.add_argument('--val_fold', type=str, help='Path to validation set', required=True)
    # hyperparams
    parser.add_argument('--batch', type=int, help='Batch size', required=False, default=32)
    parser.add_argument('--lr', type=float, help='Initial learning rate', required=False, default=1e-3)
    parser.add_argument('--epochs', type=int, help='Number of epochs', required=False, default=10)

    args = parser.parse_args()

    train_fold = args.train_fold
    val_fold = args.val_fold

    batch_size = args.batch
    learning_rate = args.lr
    n_epochs = args.epochs

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    #train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=rotation_range, horizontal_flip=True, vertical_flip=True, preprocessing_function=preprocessing_function, brightness_range=brightness_range)
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

    train_dataset = train_datagen.flow_from_directory(
        train_fold,
        target_size=(200, 200),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True,
        seed=42,
    )

    val_dataset = val_datagen.flow_from_directory(
        val_fold,
        target_size=(200, 200),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True,
        seed=42,
    )

    model_name = ''

    model = args.model

    if model == 'resnet':
        model = ResNet50(input_shape=input_shape, classes=classes)
        model_name = 'resnet'

    elif model == 'xception':
        model = Xception(input_shape=input_shape, classes=classes)
        model_name = 'xception'

    elif model == 'efficientnetb4':
        model = EfficientNetB4(input_shape=input_shape, classes=classes)
        model_name = 'efficientnetb4'

    elif model == 'cnndct':
        mean, var = welford(train_dataset)
        model = CNN_DCT(mean, var, input_shape=input_shape, classes=classes)
        model_name = 'cnndct'
    else:
        print('Model selected not valid.')

    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)

    model.compile(optimizer=opt, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=[tf.keras.metrics.AUC(), 'accuracy',])

    model.fit(
        train_dataset, 
        batch_size=batch_size, 
        epochs=n_epochs, 
        verbose="auto", 
        validation_data=val_dataset, 
        shuffle=True, 
        callbacks=[
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=factor, patience=patience_lr), 
                tf.keras.callbacks.ModelCheckpoint('models/trained_models/' + model_name +'.h5', monitor='val_accuracy', save_best_only=True), 
                tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience, restore_best_weights=True)]
        )

if __name__ == '__main__':
    main()
