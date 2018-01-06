import csv
import numpy as np
from numpy.linalg import inv
import random
import math
import sys

data = []

for i in range(18):
    data.append([])

n_row = 0
text = open('data/train.csv', 'r', encoding='big5')
row = csv.reader(text, delimiter=",")
for r in row:
    if n_row != 0:
        for i in range(3, 27):
            if r[i] != "NR":
                data[(n_row-1)%18].append(float(r[i]))
            else:
                data[(n_row-1)%18].append(float(0))
    n_row = n_row + 1
text.close()

x = []
y = []
# 每12个月
for i in range(12):
    # 一个月取连续10小时的data可以有471笔
    for j in range(471):
        x.append([])
        # 18种污染物
        for t in range(18):
            # 连续9小时
            for s in range(9):
                x[471*i+j].append(data[t][480*i+j+s])
        y.append(data[9][480*i+j+9])
x = np.array(x)
y = np.array(y)

# add square term
# x = np.concatenate((x, x**2), axis=1)

# add bias
x = np.concatenate((np.ones((x.shape[0],1)), x), axis=1)

### init weight & other hyperparams
w = np.zeros(len(x[0]))
l_rate = 10
repeat = 10000

### check your ans with close form solution
# w = np.matmul(np.matmul(inv(np.matmul(x.transpose(), x)), x.transpose()), y)

### start training
x_t = x.transpose()
s_gra = np.zeros(len(x[0]))
for i in range(repeat):
    hpyo = np.dot(x, w)
    loss = hpyo - y
    cost = np.sum(loss**2) / len(x)
    cost_a = math.sqrt(cost)
    gra = np.dot(x_t, loss)
    s_gra += gra**2
    ada = np.sqrt(s_gra)
    w = w - l_rate*gra/ada
    print('iteration: %d | Cost: %f' % (i, cost_a))

### save/read model
# save model
np.save('model.npy', w)
# read model
w = np.load('model.npy')

### read testing data
test_x = []
n_row = 0;
text = open('data/test.csv', "r")
row = csv.reader(text, delimiter=",")

for r in row:
    if n_row % 18 == 0:
        test_x.append([])
        for i in range(2, 11):
            test_x[n_row//18].append(float(r[i]))
    else:
        for i in range(2, 11):
            if r[i] != "NR":
                test_x[n_row//18].append(float(r[i]))
            else:
                test_x[n_row//18].append(float(0))
    n_row = n_row + 1
text.close()
test_x = np.array(test_x)

# add square term
# test_x = np.concatenate((test_x, test_x**2), axis=1)

# add bias
test_x = np.concatenate((np.ones((test_x.shape[0], 1)), test_x), axis=1)

### get ans.csv with your model
ans = []
for i in range(len(test_x)):
    ans.append(["id_"+str(i)])
    a = np.dot(w, test_x[i])
    ans[i].append(a)

filename = "result/predict.csv"
text = open(filename, "w+")
s = csv.writer(text, delimiter=',', lineterminator='\n')
s.writerow(["id", "value"])
for i in range(len(ans)):
    s.writerow(ans[i])
text.close()
