import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class multivar_lin_regression(object):

    def __init__(self, features, data_loc, train_size):
        self.features = features
        self.test_data, self.train_data = self.__prepare_data(data_loc, train_size)


    def __prepare_data(self, data_loc, train_size):
        data = pd.read_csv(data_loc, usecols=self.features)

        train_data = data.sample(frac=train_size)
        test_data = data[~data.index.isin(train_data.index.values)]

        return test_data, train_data

    def __covariance(self, x, mean_x, y, mean_y):
        covar = 0.0
        for i in range(len(self.train_data.index)):
            covar += (x.iloc[i] - mean_x) * (y.iloc[i] - mean_y)
        return covar

    def __coefficients(self):
        x = self.train_data['Mileage']
        y = self.train_data['Price']

        # Calculate mean and variance
        mean_x, mean_y = self.__mean(x.values), self.__mean(y.values)
        var_x, var_y = self.__variance(x.values, mean_x), self.__variance(y.values, mean_y)

        b1 = self.__covariance(x, mean_x, y, mean_y) / var_x
        b0 = mean_y - b1 * mean_x
        #print(b1, b0)
        return b0, b1

    def __variance(self, values, mean):
        return sum([(x-mean)**2 for x in values])

    def __mean(self, values):
        return sum(values) / float(len(values))

    def regress(self):        
        # calculate coefficients
        self.b0, self.b1 = self.__coefficients()

        self.test_data['predictions'] = self.test_data.index

        self.__simple_linear_regression()

        print(self.__rmse_metric())

        self.test_data.plot(x='Mileage', y='Price', kind='scatter')
        self.test_data.plot(x='Mileage', y='predictions', kind='scatter', color='orange')
        plt.show()
        
    def __simple_linear_regression(self):
        for i in range(len(self.test_data)):
            self.test_data.iloc[i, 2] = self.b0 + self.b1 * self.test_data.iloc[i, 0]

    def __rmse_metric(self):
        sum_error = 0.0

        for i in range(len(self.test_data.index)):
            prediction_error = self.test_data.iloc[i, 2] - self.test_data.iloc[i, 1]
            sum_error += (prediction_error ** 2)
        mean_error = sum_error / float(len(self.test_data.index))

        return np.sqrt(mean_error)


def main():
    #instantiate class <-- add , 'Liter'
    features = ['Price', 'Mileage']
    regressor = multivar_lin_regression(features, './car_data.csv', 0.8)
    regressor.regress()



if __name__ == '__main__':
    main()