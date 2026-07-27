import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class predict:

    def __init__(self, x, m=0, b=0):
        self.x = x
        self.m, self.b = self._load_variables(m, b)

        # read data 
        self.df = pd.read_csv("data.csv")

        # graph data
        plt.scatter(self.df['km'], self.df['price'], color="blue")
        plt.xlabel("km")
        plt.ylabel("price")
        plt.grid()
        plt.title("Prix des voitures en fonction du kilométrage")

    def _load_variables(self, default_m, default_b):
        variables_path = Path("variables")
        if not variables_path.exists():
            return default_m, default_b

        content = variables_path.read_text().strip().split()
        if len(content) < 2:
            return default_m, default_b

        return float(content[0]), float(content[1])

    def predict(self, x):
        y = self.m * x + self.b 

        return y

def main():
    km_value = float(input("Enter the mileage in kilometers: "))
    lr = predict(km_value)
    y = lr.predict(km_value)
    print(f"Predicted price for {km_value} km: {y}")

    plt.scatter([lr.x], [y], color="red")
    plt.savefig("price.png")

if __name__ == "__main__":
    main()