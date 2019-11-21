import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import neighbors, naive_bayes, linear_model
from sklearn.metrics import classification_report as report



class Classifier:
    def __init__(self, path):
        self.path = path

    def getdata(self):
        with open(self.path)as file:
            data = pd.read_csv(file, sep=';')
        return data

    def preprocess(self, data):
        pd.options.mode.chained_assignment = None
        copy = data.copy()
        for column in data.columns:
            if column == 'y':
                for j in range(data.shape[0]):
                    copy[column][j] = 0 if data[column][j] == 'no' else 1
                break
            if not is_numeric_dtype(data[column]):
                unique_values = list(data[column].unique())
                for j in range(data.shape[0]):
                    copy[column][j] = unique_values.index(data[column][j]) / (len(unique_values) - 1)
            else:
                Max = max(data[column])
                Min = min(data[column])
                for j in range(data.shape[0]):
                    copy[column][j] = (data[column][j] - Min) / (Max - Min)
        return copy

    def in_out_split(self, data):
        return np.array(data.iloc[:, :-1]), np.array(data.iloc[:, -1]).astype('int')

    def init_clfs(self):
        KNN_obj = neighbors.KNeighborsClassifier(n_neighbors=10, p=2, weights='distance')
        NB_obj = naive_bayes.MultinomialNB()
        LR_obj = linear_model.LogisticRegression(solver='lbfgs', max_iter=1e5)
        return KNN_obj, NB_obj, LR_obj

    def fit_data(self, objs, X_train, y_train):
        for i in range(len(objs)):
            objs[i][0].fit(X_train, y_train)

    def test_one(self, data, X_test, y_test, objs):
        with open('test data.txt', mode='w') as f:
            for i in range(10):
                f.write('{} {}\n'.format(X_test[i], y_test[i]))

        print('-- insert your test point (must have 20 attributes):')
        point = []

        """ // if you just want to copy & paste the whole point:
        point = input('-- insert your test point (must have 20 attributes): - ').split()
        point = [float(x) for x in point]
        """

        for i in range(data.shape[1] - 1):
            point.append(float(input('-- {}= '.format(data.columns[i]))))

        for obj in objs:
            y_pred = obj[0].predict([np.array(point)])
            print('--', obj[1], 'predicts output =', y_pred[0])

    def test_more(self, X_test, y_test, objs):
        n = int(input('-- how many parts do you want to split test data into?: - '))
        r = input('-- ratios of them (eg: 1:3:6 if n=3): - ').split(':')
        r = [int(x) for x in r]
        sum_r = 0

        for i in range(n):
            size = round(r[i] / (10 - sum_r) * len(X_test))
            X_test_small, y_test_small, = X_test[:size], y_test[:size]
            X_test, y_test = X_test[size:], y_test[size:]
            sum_r += r[i]

            for obj in objs:
                y_pred = obj[0].predict(X_test_small)
                print('--', obj[1])
                print(report(y_test_small, y_pred))

        """ // if you just want to test with an arbitary amount of test data:
        test_size = float(input('-- test size = ? (eg: 0.1 /...): - '))
        size = round(test_size * len(X_test))
        for obj in objs:
            y_pred = obj[0].predict(X_test[:size])
            print('--', obj[1])
            print(report(y_test[:size], y_pred))
        """

    def main(self):
        data = self.getdata()
        data = self.preprocess(data)

        X, y = self.in_out_split(data)

        KNN_obj, NB_obj, LR_obj = self.init_clfs()
        objs = [[KNN_obj, 'K-NN'], [NB_obj, 'Naive Bayes'], [LR_obj, 'Logistic Reg']]

        while(True):
            choice = input('Classify for only one point or more? (one/more): - ')

            while(True):
                ratio = input('- ratio of train and test? (train:test): - ').split(':')
                X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=int(ratio[0])/10, random_state=1)
                self.fit_data(objs, X_train, y_train)

                while(True):
                    if choice == 'one':
                        self.test_one(data, X_test, y_test, objs)
                    else:
                        self.test_more(X_test, y_test, objs)

                    retest = input('-- test again? (y/n): - ')
                    if retest == 'y':
                        continue
                    else:
                        break

                retrain = input('- train with another ratio? (y/n): - ')
                if retrain == 'y':
                    continue
                else:
                    break

            restart = input('Successfully compiled! Do you want to continue?(y/n): - ')
            if restart == 'y':
                continue
            else:
                break



if __name__ == "__main__":
    CLF = Classifier('bank-additional-full.csv')
    CLF.main()