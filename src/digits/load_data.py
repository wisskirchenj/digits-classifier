from numpy import ndarray
import keras
from sklearn.model_selection import train_test_split


class DataSet:
    def __init__(self, features: ndarray, targets: ndarray, test_size=0.3, random_state=40):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            features, targets, test_size=test_size, random_state=random_state)


def load_data(seed) -> DataSet:
    features, targets = keras.datasets.mnist.load_data()[0]
    features, targets = features[:6000], targets[:6000]
    features = features.reshape(features.shape[0], features.shape[1] * features.shape[2])
    return DataSet(features, targets, random_state=seed)
