import tensorflow as tf
from deepstack.ensemble import StackEnsemble
import argparse
from sklearn import metrics
import numpy as np
from tensorflow import keras
from keras.models import load_model
from models.cnndct import DCTLayer
from glob import glob

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--test_fold', type=str, help='Path to test set', required=True)

    args = parser.parse_args()
    
    test_fold = args.test_fold

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

    test_dataset = test_datagen.flow_from_directory(
        test_fold,
        target_size=(200, 200),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=1,
        shuffle=True,
        seed=42,
    )

    # Loading base-learners

    base_learners = []

    for filename in glob('models/trained_models/*'):

        custom_objects = None

        if 'cnndct' in filename:
            custom_objects = {'DCTLayer':DCTLayer}
        
        base_learner = load_model(filename, custom_objects=custom_objects)

        base_learners.append(base_learner)

    ensemble = StackEnsemble()

    ensemble = ensemble.load('./trained_ensemble')

    # Loaing meta-learner

    meta_learner = ensemble.model

    y_pred_base = []

    labels = []

    for i in range(test_dataset.__len__()):

        X_y = test_dataset.__getitem__(i)

        label = X_y[1].reshape(-1)[1]

        labels.append(label)

        image = X_y[0]

        prediction = []

        for base_learner in base_learners:
            base_prediction = base_learner.predict(image)
            prediction.append(base_prediction)

        prediction = np.array(prediction)
        prediction = prediction.reshape(prediction.shape[1], prediction.shape[0] * prediction.shape[2])

        y_pred_base.append(prediction)

    y_pred_base = np.array(y_pred_base)

    # reshape(-1, 2 * n_base_learners)

    y_pred_base = y_pred_base.reshape(-1,8)

    y_pred = meta_learner.predict(y_pred_base)

    # Thresholding
    # 1 for real images, 0 for generated images

    T = 0.5

    y_pred[y_pred > T] = 1
    y_pred[y_pred < T] = 0

    labels = np.array(labels)

    acc = metrics.accuracy_score(labels, y_pred)

    print('Accuracy ensemble: ', acc)

    for base_learner in base_learners:
        loss, AUC, accuracy = base_learner.evaluate(test_dataset)
        print('Accuracy base learner: ', accuracy)

if __name__ == '__main__':
    main()
