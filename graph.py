import copy
import re

import numpy as np


class Graph:
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

    def _calculate_graph(self):
        """self.graph_2d = [[abs(a[1][0] - b[1][0]) + abs(a[1][1] - b[1][1])
                          for a in self.graph_dict.items()]
                         for b in self.graph_dict.items()]"""

        X = [x[1][0] for x in self.graph_dict.items()]
        Y = [y[1][1] for y in self.graph_dict.items()]

        self.graph_2d = np.zeros([self.size, self.size])  # Шаблон матрицы относительных расстояний между пунктами

        for i in np.arange(0, self.size, 1):
            for j in np.arange(0, self.size, 1):
                if i != j:
                    self.graph_2d[i][j] = abs(X[i] - X[j]) + abs(Y[i] - Y[j])  # Заполнение матрицы
                else:
                    self.graph_2d[i][j] = float('inf')  # Заполнение главной диагонали матрицы

        self.X = X
        self.Y = Y

    def read_graph(self):
        with open(self.file, 'r', encoding='utf-8') as f:
            self._create_graph_dict(f)
            self._calculate_graph()

    def get_paths(self):
        paths = []
        S = []

        for i in range(0, 1):
            print(i)
            path, s = self.find_path_Kmean()
            S.append(s)
            paths.append(path)
            self._del_path_to_graph(path)

        self.save_tree(paths, S)

        return paths

    @staticmethod
    def print_matrix(matrix):
        print("---------------")
        for i in range(len(matrix)):
            print(matrix[i])
        print("---------------")

    def find_path_Kmean(self, shift=0):

        way = []
        matrix = copy.deepcopy(self.graph_2d)
        start = 0
        way.append(start)
        flag_bad = False

        for i in range(1, self.size):
            s = []

            for j in range(0, self.size):
                s.append(matrix[way[i - 1]][j])

            if shift > 0 and shift == i:
                s[s.index(min(s))] = float('inf')

            if s.index(min(s)) in way:
                flag_bad = True
                break

            way.append(s.index(min(s)))

            # Индексы пунктов ближайших городов соседей
            for j in range(0, i):
                matrix[way[i]][way[j]] = float('inf')
                matrix[way[j]][way[i]] = float('inf')

        if flag_bad:
            return self.find_path_Kmean(shift + 1)

        if len(set(way)) != self.size:
            print('\n_____________________\n')
            print(len(set(way)))

        S = sum([abs(self.X[way[i]] - self.X[way[i + 1]]) + abs(self.Y[way[i]] - self.Y[way[i + 1]])
                 for i in np.arange(0, self.size - 1, 1)]) + \
            (abs(self.X[way[self.size - 1]] - self.X[way[0]]) + abs(self.Y[way[self.size - 1]] - self.Y[way[0]]))

        print("WAY - ", S)
        print(len(set(way)))
        return way, S

    def _del_path_to_graph(self, path):
        for i_town, town in enumerate(path):
            if i_town != len(path) - 1:
                self.graph_2d[town][path[i_town + 1]] = float('inf')
                self.graph_2d[path[i_town + 1]][town] = float('inf')
            else:
                self.graph_2d[town][0] = float('inf')
                self.graph_2d[0][town] = float('inf')

    def save(self, paths, S):
        with open('Averina_{}.txt'.format(self.size), 'a') as f:
            f.write('\n\n')
            f.write('n = ' + str(self.size))
            f.write('\nМаршруты коммивояжёра: \n')
            for path in paths:
                for town in path:
                    f.write(str(town + 1) + ' ')
                f.write('\n')

            sum = 0
            for i, s in enumerate(S):
                sum += s
                if i != 0:
                    f.write(' + ')
                f.write(str(s))

            f.write(' = ')
            f.write(str(sum))

    def save_tree(self, paths, S):
        with open('Averina_{}.txt'.format(self.size), 'a') as f:
            f.write(f'\nc  Вес дерева = {S[0]}, число листьев = 2,\n')
            f.write(f'p edge = {self.size} {self.size - 1}\n')
            for path in paths:
                for idx, town in enumerate(path):
                    if idx != self.size - 1:
                        f.write('e ' + str(town + 1) + ' ' + str(path[idx + 1] + 1) + '\n')
