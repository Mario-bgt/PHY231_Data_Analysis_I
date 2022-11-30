import numpy as np


def wurf(n):
    chance = 0
    res = np.random.randint(1, 7, (1, n, 3))
    for i in res[0]:
        print(i)
        if 7 in i:
            print(i)
        if 1 in i and 5 in i:
            chance += 1
    print(chance/n)


wurf(1000)
