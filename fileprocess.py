import numpy as np
"""
def gen_line(filename):
    for line in open(filename):
        yield line

lines = gen_line("alpha_shape_points.csv")
profile = []
for line in lines:
    if line.strip() == "group1":
        temp = []
        for _line in lines:
            if _line.strip() == "group2":
                profile.append(temp)
                break
            else:
                temp.append(_line.strip().split(','))
"""

filename_alpha = "alpha_shape_points.csv"

def read_alpha_shape(filename):
    profile = {}
    prev_line = []
    with open(filename) as fin:
        for line in fin:
            if line.strip() == "group1":
                num = int(prev_line.strip())
                temp = []
                for _line in fin:
                    if _line.strip() == "group2":
                        profile[num] = np.array(temp, dtype=float)
                        break
                    temp.append(_line.strip().split(','))
            prev_line = line
    return profile

filename_graph = "relocated_points_all.csv"

def read_graph(filename):
    profile = {}
    with open(filename) as fin:
        num = int(fin.readline().strip())
        temp = []
        for line in fin:
            if line.strip() == "group1":
                profile[num] = []
                for _line in fin:
                    if _line.strip() == "group2":
                        profile[num].append(np.array(temp, dtype=float))
                        temp.clear()
                        for __line in fin:
                            if len(__line.strip().split(',')) == 1:
                                profile[num].append(np.array(temp, dtype=float))
                                temp.clear()
                                num = int(__line.strip())
                                break
                            temp.append(__line.strip().split(','))
                        break
                    temp.append(_line.strip().split(','))
        profile[num].append(np.array(temp, dtype=float))
    return profile



"""
point_data = {}
for root, libs, files in os.walk("outer_points"):
    for file in files:
        with open(root+'/'+file, encoding='UTF-8') as fin:
            content = list(csv.reader(fin))
            point_data[int(file[5:7])] = np.array(content[2:], dtype=int)[2:,:3]

for item in point_data.items():
    print(item)
"""

if __name__ == "__main__":
    alpha = read_alpha_shape(filename_alpha)
    graph = read_graph(filename_graph)

    """ 
    for item in graph.items():
        print(item[1][0].min(0),item[1][0].max(0))
    """

    index = np.array([0, 1])
    lib = np.arange(0,9).reshape(3,3)

    print(lib)
    print(lib[tuple(index)])
    print(np.subtract([[1,1],[2,2],[3,3]],[1,1]))
    print(np.vstack(([1,2],[3,4])))
    print(5//2)