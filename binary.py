import copy
import re
import sys

import numpy as np


class Binary:
    def __init__(self, file_name=None):
        if not file_name:
            return False

        self.size = None
        self.file = file_name
        self.graph_2d = None
        self.graph_dict = {}
        self.leng = 0

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

    def search_min(self, vizited):  # 1 место для оптимизации
        min = max(self.graph_2d)
        for ind in vizited:
            for index, elem in enumerate(self.graph_2d[ind]):
                if 0 < elem < min and index not in vizited:
                    min = elem  # веса путей
                    index2 = index  # индекс города
        return [min, index2]

    def prim(self):
        L = int(self.size / 10)
        const_bin = 3
        toVisit = [i for i in range(1, self.size)]  # города кроме начального(0)
        vizited = [0]
        result = [0]  # начнем с минска
        for index in toVisit:
            weight, ind = self.search_min(vizited)
            result.append(weight)  # в результат будут заноситься веса
            vizited.append(ind)  # содержит карту пути
        return result

    def bor(self):
        INF = float('inf')
        L = int(self.size / 10)
        CONST_BIN = 3

        dist = [0, ]
        used = [False] * self.size
        used[0] = True
        ans = 0

        for i in range(self.size):
            min_dist = INF

            for j in range(self.size):
                if (not used[j]) and (self.graph_2d[i][j] < min_dist):
                    min_dist = self.graph_2d[i][j]
                    u = j
                    print(j)

            ans += min_dist

            used[u] = True
            dist.append(u)
            """for v in range(self.size):
                dist[v] = min(dist[v], self.graph_2d[u][v])"""

        return dist, len(set(dist)), ans

    @property
    def boruvka_algorithm(self):
        """
        Алгоритм Борувки для поиска минимального остовного дерева
        1. каждая вершина - дерево
        2. добавляем минимальное ребро к каждому дереву
        3. повторяем, пока не останется одно дерево
        Алгоритм состоит из нескольких шагов:

        Изначально каждая вершина графа G— тривиальное дерево, а ребра не принадлежат никакому дереву.
        Для каждого дерева T найдем минимальное инцидентное ему ребро. Добавим все такие ребра.
        Повторяем шаг 2 пока в графе не останется только одно дерево T.
        :return:
        """
        full_len = 0

        def vert_search(vert, founded):
            prev = None
            if vert in founded:
                return founded
            if len(founded) > 0:
                prev = founded[-1]
            founded.append(vert)
            if prev:
                self.leng += self.graph_2d[prev][vert]
            for v in tree_dict[vert]:
                founded = vert_search(v, founded)

            return founded

        def check_vert(it, i):
            e = tree_dict[it][i]
            tree_dict[it].pop(i)
            for idx in range(self.size):
                path = vert_search(idx, set())
                if len(path) == self.size:
                    marked.append(i)
                    tree_dict[it].update({i: e})
                    return -1
            tree_dict[it].update({i: e})
            return 0

        def found_available(vert, neigh):
            available_table[vert][neigh] = True
            available_table[neigh][vert] = True
            n_list = [i for i in range(len(available_table[neigh])) if available_table[neigh][i]
                      and i != vert and not available_table[vert][i]]
            for i in n_list:
                found_available(vert, i)

        def find_av(vert, neigh):
            if vert in tree_dict[neigh]:
                return True
            for i in tree_dict[neigh]:
                if find_av(vert, i):
                    return True
            return False

        available_table = [[] * self.size for i in range(self.size)]
        L = int(self.size / 10)
        vert_degree = 3
        marked = []
        # tree_dict = {i: {self.graph_2d[i].tolist().index(min(self.graph_2d[i])): min(self.graph_2d[i])} for i in range(self.size)}
        tree_dict = {i: dict() for i in range(self.size)}
        blacklist = [[i, ] for i in range(self.size)]
        # print(f'TREE::{len(tree_list)}\n\nDICT::{tree_dict}')
        tree = False
        added = set()
        while not tree:
            list_for_add = []
            for item in tree_dict:
                neighbours = {i: val for i, val in enumerate(self.graph_2d[item])
                              if i not in tree_dict[item] and i not in added and i != item
                              and item not in tree_dict[i] and i not in blacklist[item]}
                # print(f'CLOSE::{item}_{neighbours}\n')
                close_neigh_list = [i for i in neighbours if neighbours[i] == min(neighbours.values())]
                if close_neigh_list:
                    closest = min(close_neigh_list)
                    if len(tree_dict[item]) < 3:
                        # tree_dict[item].update({closest: self.graph_2d[item][closest]})
                        list_for_add.append([item, closest, self.graph_2d[item][closest], None])
            for i in list_for_add:
                for j in list_for_add:
                    if (i[3] is None and j[3] is None and j[1] == i[0] and i[1] == j[0]) or (j[1] == i[0] and j[0] in tree_dict[i[0]]):
                        j[3] = False

                if i[3] is None:
                    for j in list_for_add:
                        if i != j and i[1] == j[1] and j[3] is None:
                            if i[2] <= j[2]:
                                j[3] = False
                            else:
                                i[3] = False
                                break

            # print(f'ITEMS::{list_for_add}')
            for i in tree_dict:
                for j in list_for_add:
                    if j[1] == i and j[0] in tree_dict[i]:
                        j[3] = False

            # print(f'LIST::{list_for_add}\n')
            for item in list_for_add:
                if item[3] is None:
                    tree_dict[item[0]].update({item[1]: item[2]})
                    if item[0] == 51:
                        print(f'ITEM::{tree_dict[item[0]]}\nFIND::{find_av(item[0], item[0])}')
                    if find_av(item[0], item[0]):
                        item[3] = False
                        tree_dict[item[0]].pop(item[1])
                        blacklist[item[0]].append(item[1])
                        continue
                    else:
                        added.add(item[1])
                else:
                    blacklist[item[0]].append(item[1])

            # print(f'ADDED::{added}_{len(added)}\n\n')
            # print(f'DICT2::{tree_dict}\n\n')
            # print(f'AVAIL::{available_table}\n\n')

            input()
            # added += set([item[1] for item in list_for_add if item[3]])
            if len(added) == self.size - 1:
                tree = True
            """
            по только добавленным вершинам в деревья выбираем наименьшее ребро для каждой, остальные удаляем
            """

        print('RES::{}'.format("\n".join([str(item) + ':' + str(tree_dict[item]) for item in tree_dict])))
        print('PATH::{}'.format('\n'.join([re.sub(r"[]{[}]", "", str({item: [*tree_dict[item]]})) for item in tree_dict if tree_dict[item]])))
        print(f'SUMPATH::{sum([sum(list(tree_dict[item].values()) )for item in tree_dict if tree_dict[item]])}')


    def get(self):
        print(self.bor())
        return

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
