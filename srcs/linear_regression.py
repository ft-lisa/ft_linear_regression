from dbm import error
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class linear_regression:

    def __init__(self):
        self.m = 0
        self.b = 0


        # read data 
        self.df = pd.read_csv("data.csv")

        # graph data
        plt.scatter(self.df['km'], self.df['price'], color="blue", label="data points")
        plt.xlabel("km")
        plt.ylabel("price")
        plt.grid()
        plt.title("Prix des voitures en fonction du kilométrage")

    def predict(self, x):
            y = self.m * x + self.b    

            return y

    
    def train(self):
        km = self.df['km']
        price = self.df['price']


        # data normalization
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
                self.real_m = self.m * self.price_std / self.km_std
                self.real_b = self.price_mean + self.b * self.price_std - self.real_m * self.km_mean

                Path("variables").write_text(f"{self.real_m} {self.real_b}")

                # graph regression line
                x_values = np.linspace(self.df['km'].min(), self.df['km'].max(), 100)
                y_values = self.real_m * x_values + self.real_b
                plt.plot(x_values, y_values, color="red", label="regression line")
                plt.legend()
                plt.savefig("linear_regression.png")
                break
            previous_loss = loss
        
def main():
    lr = linear_regression()
    lr.train()

if __name__ == "__main__":
    main()