import numpy as np
import fileprocess
import matplotlib.pyplot as plt

def shift_index(points_as_indexes, bias, scaling):
    index = points_as_indexes * scaling

    #bias = np.repeat([[95, 60]], points.shape[0], axis=0)
    return np.add(index.astype(int), bias)

def graph_add_window(length, height, graphs, alpha_graphs):
    g_temp = []
    l_temp = []

    for number, graph in graphs.items():
        points = np.concatenate((graph[0], graph[1]), axis=0)

        indexes = shift_index(points,[-95, 60], 10) # points [95~205， -60~60]-->[0~110,0~120]
        full_graph = np.zeros((110, 120))
        for index in indexes:
            full_graph[tuple(index)] = 1

        alpha_graph = shift_index(alpha_graphs[number], [-95, 60], 10)
        label_graph = np.zeros((110,120), dtype=int)
        for point in alpha_graph:
            label_graph[tuple(point)] = 1

        zeros = np.zeros((length, height), dtype=int)
        ones = np.ones((length, height), dtype=int)

        for x in range(110 - length):
            for y in range(120 - height):
                temp = np.array(full_graph[x: x + length, y: y + height])
                if((temp == zeros).any() or (temp == ones).any()):
                    g_temp.append(temp[:, :, np.newaxis])
                    l_temp.append(label_graph[x + length//2, y + height//2])
        if number > 45: break

    pic = np.array(g_temp)
    label = np.array(l_temp)
    print(pic.shape)
    print(label.shape)

    fig, axes = plt.subplots(3, 3, sharex="col", sharey="row", figsize=(9, 9))
    for _num, _axe in enumerate(axes):
        for num, axe in enumerate(_axe):
            axe.imshow(np.squeeze(pic[2000 + 100*(_num*3+num)],2), cmap=plt.get_cmap("jet"))
            axe.set_title("No. %d"%(2000 + 100*(_num*3+num)))
    plt.show()
    return pic, label

def single_graph_add_window(length, height, graph):
    g_temp = []

    points = np.concatenate((graph[0], graph[1]), axis=0)
    indexes = shift_index(points, [-95, 60], 10)  # points [95~205， -60~60]-->[0~110,0~120]
    full_graph = np.zeros((110 + length, 120 + height))
    for index in indexes:
        full_graph[(index[0]+length//2, index[1]+height//2)] = 1

    for x in range(110):
        for y in range(120):
            temp = np.array(full_graph[x: x + length, y: y + height])
            g_temp.append(temp[:, :, np.newaxis])

    pic = np.array(g_temp)
    print(pic.shape)
    # plt.imshow(np.squeeze(pic[4803],2), cmap=plt.get_cmap("jet"))
    # plt.show()
    return pic, full_graph[length//2 : length//2+110, height//2: height//2+120]

def single_alpha(alpha):
    indexes = shift_index(alpha, [-95, 60], 10)  # points [95~205， -60~60]-->[0~110,0~120]
    full_graph = np.zeros((110, 120))
    for index in indexes:
        full_graph[tuple(index)] = 1
    return full_graph

if __name__ == "__main__":
    graphs = fileprocess.read_graph("relocated_points_all.csv")
    alpha_graph = fileprocess.read_alpha_shape("alpha_shape_points.csv")
    graph_add_window(35, 35, graphs, alpha_graph)