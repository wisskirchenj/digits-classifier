from digits.load_data import DataSet
from sklearn.preprocessing import Normalizer


def preprocess(data_set: DataSet):
    normalizer = Normalizer()
    data_set.x_train = normalizer.fit_transform(data_set.x_train)
    data_set.x_test = normalizer.transform(data_set.x_test)
