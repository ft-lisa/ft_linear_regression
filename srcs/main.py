from linear_regression import *

def main():
    lr = linear_regression(300000, 0, 0)
    y = lr.predict(300000)
    plt.scatter([lr.x], [y], color="red")
    plt.savefig("graph1.png")

    lr.train()
    y = lr.predict_real(300000)
    plt.scatter([lr.x], [y], color="red")
    plt.savefig("graph2.png")




if __name__ == "__main__":
    main()
