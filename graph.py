import random

import numpy as np
import re
import matplotlib.pyplot as plt
from numpy import exp, sqrt


class Graph():
    def __init__(self, file_name=None):
        if not file_name:
            return False
        self.size = None
        self.file = file_name
        self.graph_2d = None
        self.graph_dict = {}

    def _create_graph_dict(self, file):
        self.size = int(re.search(r'\d+', file.readline()).group())
        for line in file.readlines():
            vert = list(map(int, re.findall(r'\d+', line)))
            self.graph_dict.update({vert[0]: [vert[1], vert[2]]})
        print(self.graph_dict)

    def _calculate_graph(self):
        self.graph_2d = [[abs(a[1][0] - b[1][0]) + abs(a[1][1] - b[1][1])
                          for a in self.graph_dict.items()]
                         for b in self.graph_dict.items()]

        """for i in range(self.size):
            self.graph_2d[i][i] = float('inf')"""
        print(self.graph_2d)

    def read_graph(self):
        with open(self.file, 'r', encoding='utf-8') as f:
            self._create_graph_dict(f)
            self._calculate_graph()

    # Функция нахождения минимального элемента, исключая текущий элемент
    def Min(self, lst, myindex):
        return min(x for idx, x in enumerate(lst) if idx != myindex)

    # функция удаления нужной строки и столбцах
    def Delete(self, matrix, index1, index2):
        del matrix[index1]
        for i in matrix:
            del i[index2]
        return matrix

    def get_paths(self):
        paths = []
        for i in range(0, 3):
            print(i)
            # https://habr.com/ru/post/329604/
            # https://github.com/Clever-Shadow/python-salesman/blob/master/salesman.py
            path = self.find_path_Kmean()
            print(path)
            paths.append(path)
            self._del_path_to_graph(path)

        return self._paths_to_format(paths)

    def print_matrix(self, matrix):
        print("---------------")
        for i in range(len(matrix)):
            print(matrix[i])
        print("---------------")

    def find_path_Kmean(self):
        m = 100
        # ib = 3
        way = []
        a = 0
        X = [x[1][0] for x in self.graph_dict.items()]
        Y = [x[1][1] for x in self.graph_dict.items()]

        M = np.zeros([self.size, self.size])  # Шаблон матрицы относительных расстояний между пунктами
        for i in np.arange(0, self.size, 1):
            for j in np.arange(0, self.size, 1):
                if i != j:
                    M[i, j] = abs(X[i] - X[j]) + abs(Y[i] - Y[j])  # Заполнение матрицы
                else:
                    M[i, j] = float('inf')  # Заполнение главной диагонали матрицы

        ib = random.randint(1, self.size)
        way.append(ib)
        for i in np.arange(1, self.size, 1):
            s = []
            for j in np.arange(0, self.size, 1):
                s.append(M[way[i - 1], j])
            way.append(s.index(min(s)))  # Индексы пунктов ближайших городов соседей
            for j in np.arange(0, i, 1):
                M[way[i]][way[j]] = float('inf')
                M[way[i]] [way[j]] = float('inf')

        return way

    def drow_path(self, X, Y, way, a, m, ib):
        S = sum([sqrt((X[way[i]] - X[way[i + 1]]) ** 2 + (Y[way[i]] - Y[way[i + 1]]) ** 2) for i in
                 np.arange(0, self.size - 1, 1)]) + sqrt(
            (X[way[self.size - 1]] - X[way[0]]) ** 2 + (Y[way[self.size - 1]] - Y[way[0]]) ** 2)

        plt.title('Общий путь-%s.Номер города-%i.Всего городов -%i.\n Координаты X,Y случайные числа от %i до %i' % (
            round(S, 3), ib, self.size, a, m), size=14)
        n = self.size
        X1 = [X[way[i]] for i in np.arange(0, n, 1)]
        Y1 = [Y[way[i]] for i in np.arange(0, n, 1)]
        plt.plot(X1, Y1, color='r', linestyle=' ', marker='o')
        plt.plot(X1, Y1, color='b', linewidth=1)
        X2 = [X[way[n - 1]], X[way[0]]]
        Y2 = [Y[way[n - 1]], Y[way[0]]]
        plt.plot(X2, Y2, color='g', linewidth=2, linestyle='-', label='Путь от  последнего \n к первому городу')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()

    def _del_path_to_graph(self, path):
        for i_town, town in enumerate(path):
            if i_town != len(path) - 1:
                self.graph_2d[town][path[i_town + 1]] = 0
                self.graph_2d[path[i_town + 1]][town] = 0
            else:
                self.graph_2d[town][0] = 0
                self.graph_2d[0][town] = 0

    def _paths_to_format(self, paths):
        pass