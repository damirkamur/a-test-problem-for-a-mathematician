from typing import Callable
from matplotlib import pyplot as plt
from math import exp
from random import randint
from time import time

import numpy as np


def timer(func: Callable):
    """
    Декоратор для оценки времени функции
    """

    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        print(f'Время выполнения функции {func.__name__}: ', time() - start_time, ' c.')
        return result

    return wrapper


class RungeKutta:
    """
    x0(float) - начальная точка
    x1(float) - конечная точка
    N(int) - число узлов сетки
    func(Callable) - функция, принимающая скаляр x и вектор y, возвращающая вектор
        y'(x) = func(x,y)
    y0(np.ndarray) - вектор начальных значений y
    """

    def __init__(self, x0: float, x1: float, N: int, func: Callable, y0: np.ndarray) -> None:
        self.__x0 = x0
        self.__x1 = x1
        self.__N = N
        self.__x = np.linspace(x0, x1, N)
        self.__h = (x1 - x0) / (N - 1)
        self.__y = np.vstack([y0, np.zeros((N - 1, len(y0)))])
        self.__func = func

    @timer
    def solve(self) -> None:
        for i in range(1, self.__N):
            # y_{i+1} = y_{i} + h / 6 (k1 + 2*k2 + 2*k3 + k4)
            self.__y[i] = self.__y[i - 1] + self.__h / 6 * (
                    self.__k1(i - 1) + 2 * self.__k2(i - 1) + 2 * self.__k3(i - 1) + self.__k4(i - 1))

    def __k1(self, n: int) -> np.ndarray:
        # k1 = f(xn, yn)
        return self.__func(self.__x[n], self.__y[n])

    def __k2(self, n: int) -> np.ndarray:
        # k2 = f(xn + h/2, yn + h/2 k1)
        return self.__func(self.__x[n] + self.__h / 2, self.__y[n] + self.__h / 2 * self.__k1(n))

    def __k3(self, n: int) -> np.ndarray:
        # k3 = f(xn + h/2, yn + h/2 k2)
        return self.__func(self.__x[n] + self.__h / 2, self.__y[n] + self.__h / 2 * self.__k2(n))

    def __k4(self, n: int) -> np.ndarray:
        # k4 = f(xn + h, yn + h k3)
        return self.__func(self.__x[n] + self.__h, self.__y[n] + self.__h * self.__k3(n))

    def plot(self, y_index: int, save_to_file: bool = False):
        plt.figure(randint(0, 10 ** 6))
        plt.plot(self.__x, self.__y[:, y_index])
        plt.minorticks_on()
        plt.grid(which='major', linewidth=1)
        plt.grid(which='minor', linestyle=':')
        plt.savefig(f'images/y_{y_index} - (N = {self.__N}).png', dpi=300) if save_to_file else plt.show()

    def plot_with_analit_solution(self, y_index: int, save_to_file: bool = False):
        plt.figure(randint(0, 10 ** 6))
        y_analit = np.array([self.__analit_test_function(x) for x in self.__x])
        plt.plot(self.__x, y_analit, 'r')
        plt.plot(self.__x, self.__y[:, y_index], 'c--')
        plt.minorticks_on()
        plt.grid(which='major', linewidth=1)
        plt.grid(which='minor', linestyle=':')
        plt.legend(('Аналитическое решение', 'Численное решение'))
        print(f'Невязка (N = {self.__N}) = ',
              sum([(self.__y[i, y_index] - y_analit[i]) ** 2 for i in range(len(y_analit))]))
        plt.savefig(f'images/y_{y_index} with analit solution - (N = {self.__N}).png',
                    dpi=300) if save_to_file else plt.show()

    def __analit_test_function(self, x: float) -> float:
        return -1 / 12 * exp(-3 * x) * x * (-36 - 54 * x + 16 * x ** 2 + 129 * x ** 3)


def func(x: float, y: np.ndarray) -> np.ndarray:
    coef = np.array([[0, 1, 0, 0, 0],
                     [0, 0, 1, 0, 0],
                     [0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 1],
                     [-243, -405, -270, -90, -15]])
    return coef.dot(y)


task = RungeKutta(x0=0, x1=5, N=1000, func=func, y0=[0, 3, -9, -8, 0])
task.solve()
task.plot_with_analit_solution(0, save_to_file=True)
