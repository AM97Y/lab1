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
        self.graph_2d = np.array([[abs(a[1][0] - b[1][0]) + abs(a[1][1] - b[1][1])
                                   for a in self.graph_dict.items()]
                                  for b in self.graph_dict.items()]
                                 )
        print(self.graph_2d)

    def read_graph(self):
        with open(self.file, 'r', encoding='utf-8') as f:
            self._create_graph_dict(f)
            self._calculate_graph()

    def get_paths(self):
        paths = []
        for i in range(0, 3):
            print(i)
            path = self._find_path()
            paths.append(path)
            self._del_path_to_graph(path)

        return self._paths_to_format(paths)

    def _find_path(self):
        n = self.size
        #m = 100
        ib = 3
        way = []
        a = 0
        #X = np.random.uniform(a, m, n)
        #Y = np.random.uniform(a, m, n)
        # X=[10, 10, 100,100 ,30, 20, 20, 50, 50, 85, 85, 75, 35, 25, 30, 47, 50]
        # Y=[5, 85, 0,90,50, 55,50,75 ,25,50,20,80,25,70,10,50,100]
        # n=len(X)
        M = self.graph_2d  # Шаблон матрицы относительных расстояний между пунктами
        """for i in np.arange(0, n, 1):
            for j in np.arange(0, n, 1):
                if i != j:
                    M[i, j] = sqrt((X[i] - X[j]) ** 2 + (Y[i] - Y[j]) ** 2)  # Заполнение матрицы
                else:
                    M[i, j] = float('inf')  # Заполнение главной диагонали матрицы"""
        way.append(ib)
        for i in np.arange(1, n, 1):
            s = []
            for j in np.arange(0, n, 1):
                s.append(M[way[i - 1], j])
            way.append(s.index(min(s)))  # Индексы пунктов ближайших городов соседей
            for j in np.arange(0, i, 1):
                M[way[i], way[j]] = float('inf')
                M[way[i], way[j]] = float('inf')
        S = sum([sqrt((X[way[i]] - X[way[i + 1]]) ** 2 + (Y[way[i]] - Y[way[i + 1]]) ** 2) for i in
                 np.arange(0, n - 1, 1)]) + sqrt((X[way[n - 1]] - X[way[0]]) ** 2 + (Y[way[n - 1]] - Y[way[0]]) ** 2)
        plt.title('Общий путь-%s.Номер города-%i.Всего городов -%i.\n Координаты X,Y случайные числа от %i до %i' % (
        round(S, 3), ib, n, a, m), size=14)
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
        pass

    def _paths_to_format(self, paths):
        pass
