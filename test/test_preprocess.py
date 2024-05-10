import numpy as np

from digits.load_data import DataSet
from digits.preprocess import preprocess


class TestPreprocess:

    #  The function should correctly normalize the training data using the Normalizer object.
    def test_normalize_training_data(self):
        # Arrange
        features = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        targets = np.array([0, 1, 0])
        data_set = DataSet(features, targets)
        data_set.x_train = np.array([[1, 2, 3], [4, 5, 6]])
        data_set.x_test = np.array([[7, 8, 9]])

        # Act
        preprocess(data_set)

        # Assert
        assert np.allclose(data_set.x_train,
                           np.array([[0.26726124, 0.53452248, 0.80178373], [0.45584231, 0.56980288, 0.68376346]]))
        assert np.allclose(data_set.x_test,
                           np.array([[0.50257071, 0.57436653, 0.64616234]]))
