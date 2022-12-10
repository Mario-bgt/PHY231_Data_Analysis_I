import numpy as np

my_file = open("example.txt", "r")
data = my_file.read()
data = data.replace('\n', ' ')
data = data.replace('.', '')
data_into_list = data.split(" ")
length = [len(i) for i in data_into_list]
average = np.average(length)
print(max(length))
longest = data_into_list[length.index(max(length))]
print(longest)
print(average)
my_file.close()
