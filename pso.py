import numpy as np
import random
import matplotlib.pyplot as plt

# Rosenbrock function for calculating fitness values of particles
def rosenbrock(x, y):
    return (0 - x) ** 2 + 100 * (y - x ** 2) ** 2


# Rastringin function for the same purpose
def rastrigin(x, y):
    return 20 + (x ** 2 - 10 * np.cos(2 * np.pi * x)) + (y ** 2 - 10 * np.cos(2 * np.pi * y))

# main pso algorithm that plots every nth generation where n is divisble by number of iterations divided by 5.
def pso(iter, is_rosenbrock):
    # initializing the variables
    size = 25
    dimension = 2
    a = 0.9
    b = 2
    c = 2

    # designing contour plot
    if (is_rosenbrock):
        plt.figure()
        plt.xlim(-3, 4)
        plt.ylim(-4, 5)
        X = np.linspace(-3, 4, 100)
        Y = np.linspace(-4, 5, 100)
        uX, uY = np.meshgrid(X, Y)
        Z = rosenbrock(uX, uY)
    else:
        plt.figure()
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        X = np.linspace(-5, 5, 100)
        Y = np.linspace(-5, 5, 100)
        uX, uY = np.meshgrid(X, Y)
        Z = rastrigin(uX, uY)

    plt.contourf(uX, uY, Z, cmap="BuPu")
    plt.colorbar()

    # initializing the positions and velocities arbitrarily
    x = np.random.randint(-500, 500, (size, dimension)) / 100
    v = np.random.randint(-10, 10, (size, dimension)) / 100
    pbest = x

    # finding the global best position from newly initialized positions
    for i in range(size):
        if (i == 0):
            gbest = x[i]
        else:
            if (is_rosenbrock):
                if (rosenbrock(x[i][0], x[i][1]) < rosenbrock(gbest[0], gbest[1])):
                    gbest = x[i]
            else:
                if (rastrigin(x[i][0], x[i][1]) < rastrigin(gbest[0], gbest[1])):
                    gbest = x[i]

    k = 0
    while (k < iter):

        # for each particle, updating its personal best and if necessary global best when they achieved a better position
        if (k > 0):
            for i in range(size):
                if (is_rosenbrock):
                    if rosenbrock(x[i][0], x[i][1]) < rosenbrock(pbest[i][0], pbest[i][1]):
                        pbest[i] = x[i]
                        if pbest[i] < pbest[gbest]:
                            gbest = pbest[i]
                else:
                    if rastrigin(x[i][0], x[i][1]) < rastrigin(pbest[i][0], pbest[i][1]):
                        pbest[i] = x[i]
                        if pbest[i] < pbest[gbest]:
                            gbest = pbest[i]

        # determining velocities and updating positions accordingly for each particle-dimension pair
        for i in range(size):
            for j in range(dimension):
                v[i][j] = a * v[i][j] + b * random.uniform(0, 1) * (pbest[i][j] - x[i][j]) + c * random.uniform(0,
                                                                                                                1) * (
                                      gbest[j] - x[i][j])
                x[i][j] += v[i][j]

        k += 1
        if (k % (iter / 5) == 0 and k != iter):
            plt.plot(x[:, 0], x[:, 1], marker='x', linestyle='None')

        # parameter 'a' is reduced to 0.4 over time
        if (k % int(iter / 6) == 0 and a > 0.5):
            a -= 0.1

        if k == iter:
            plt.plot(x[:, 0], x[:, 1], marker='o', linestyle='None', color='k')


if __name__ == "__main__":
    pso(100, True)  # true for rosenbrock
    pso(100, False)  # false for rastrigin
    plt.show()