from dbm import error

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class linear_regression:

    def __init__(self, x, m, b):
        self.x = x
        self.m = m
        self.b = b


        # read data 
        self.df = pd.read_csv("data.csv")

        # graph data
        plt.scatter(self.df['km'], self.df['price'], color="blue")
        plt.xlabel("km")
        plt.ylabel("price")
        plt.grid()
        plt.title("Prix des voitures en fonction du kilométrage")

    def predict(self, x):
        y = self.m * x + self.b    
        # plt.scatter([x], [y], color="red")
        
        return y

    
    def train(self):
        km = self.df['km']
        price = self.df['price']

        self.km_mean = km.mean()
        self.km_std = km.std()
        self.price_mean = price.mean()
        self.price_std = price.std()

        self.km_norm = (km - self.km_mean) / self.km_std
        self.price_norm = (price - self.price_mean) / self.price_std

        alpha = 0.1
        n = len(self.km_norm)
        previous_loss = float("inf")

        while True:
            y_pred = self.predict(self.km_norm)
            loss = ((y_pred - self.price_norm) ** 2).mean()
            error = y_pred - self.price_norm
            gradient_m = (2/n) * np.sum(error * self.km_norm)
            gradient_b = (2/n) * np.sum(error)
            self.m -= alpha * gradient_m
            self.b -= alpha * gradient_b
            if abs(previous_loss - loss) < 1e-10:
                break
            previous_loss = loss

    def predict_real(self, km_value):
        km_norm = (km_value - self.km_mean) / self.km_std
        price_norm = self.m * km_norm + self.b
        return price_norm * self.price_std + self.price_mean