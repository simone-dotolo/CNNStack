from deepstack.base import KerasMember
from deepstack.ensemble import StackEnsemble
import argparse
from glob import glob
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from models.cnndct import DCTLayer
from sklearn import metrics

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--train_fold', type=str, help='Path to training set', required=True)
    parser.add_argument('--val_fold', type=str, help='Path to validation set', required=True)

    args = parser.parse_args()

    train_fold = args.train_fold
    val_fold = args.val_fold

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

    train_dataset = train_datagen.flow_from_directory(
        train_fold,
        target_size=(200, 200),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=1,
        shuffle=True,
        seed=42,
    )

    val_dataset = val_datagen.flow_from_directory(
        val_fold,
        target_size=(200, 200),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=1,
        shuffle=True,
        seed=42,
    )

    members = []

    for filename in glob('models/trained_models/*'):

        if filename.endswith('.h5'):
            model_name = filename.split('.')[0]
            model_name = model_name.split('/')[2]

        custom_objects = None

        if 'cnndct' in filename:
            custom_objects = {'DCTLayer':DCTLayer}
        
        model = load_model(filename, custom_objects=custom_objects)

        train_batch = train_dataset
        val_batch = val_dataset

        member = KerasMember(name=model_name, keras_model=model, train_batches=train_batch, val_batches=val_batch)

        members.append(member)

    ensemble = StackEnsemble()

    for member in members:
        ensemble.add_member(member)

    ensemble.fit()

    print(ensemble.describe())

    ensemble.save('./trained_ensemble/')

if __name__ == '__main__':
    main()
