import time
from digits.analyze import Analyzer
from digits.load_data import load_data, DataSet
from digits.preprocess import preprocess

SEED = 40


class DigitClassifier:

    @staticmethod
    def output(results: dict):
        for result in results:
            print(f'{result}\nbest estimator: {results[result][0]}\naccuracy: {results[result][1]:.3f}\n')

    def main(self):
        start_time = time.time()
        data_set: DataSet = load_data(SEED)
        preprocess(data_set)
        results = Analyzer(data_set, SEED).analyze()
        self.output(results)
        end_time = time.time()
        print(f"Runtime of the program is {end_time - start_time} seconds")


if __name__ == '__main__':
    DigitClassifier().main()
