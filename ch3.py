##################### One input - one output

weight = 0.1

def neural_network(input, weight):
    prediction = input * weight
    return prediction

number_of_toes = [8.5, 9.5, 10, 9]
input = number_of_toes[0]
pred = neural_network(input, weight)
print(pred)


#################### Multi input - one output

def w_sum(a, b):
    assert len(a) == len(b)
    return sum(p * q for p, q in zip(a, b))

weights = [0.1, 0.2, 0]

def neural_network(input, weights):
    pred = w_sum(input, weights)
    return pred

toes = [8.5, 9.5, 9.9, 9.0]
wlred = [0.65, 0.8, 0.8, 0.9]
nfans = [1.2, 1.3, 0.5, 1.0]

input = [toes[0], wlred[0], nfans[0]]

pred = neural_network(input, weights)
print(pred)


##################### Multi input - one output [in numpy]
import numpy as np

weights = np.array([0.1, 0.2, 0.0])

def neural_network(input, weights):
    pred = input @ weights
    return pred

toes = np.array([8.5, 9.5, 9.9, 9.0])
wlred = np.array([0.65, 0.8, 0.8, 0.9])
nfans = np.array([1.2, 1.3, 0.5, 1.0])

input = np.array([toes[0], wlred[0], nfans[0]])
pred = neural_network(input, weights)
print(pred)


###################### One input - multi output

def ele_mul(c, a):
    return [p * q for p, q in zip([c] * len(a), a)]

weights = [0.3, 0.2, 0.9]

def neural_network(input, weights):
    pred = ele_mul(input, weights)
    return pred

wlrec = [0.65, 0.8, 0.8, 0.9]
input = wlrec[0]
pred = neural_network(input, weights)
print(pred)


###################### Multi input - multi output

def vect_mat_mul(vect, matrix):
    return [w_sum(vect, m) for m in matrix]

weights = [
#   #toes %win #fans
    [0.1, 0.1, -0.3], # hurt
    [0.1, 0.2, 0.0], # win
    [0.0, 1.3, 0.1], # sad
]


def neural_network(input, weights):
    pred = vect_mat_mul(input, weights)
    return pred

toes = [8.5, 9.5, 9.9, 9.0]
wlrec = [0.65,0.8, 0.8, 0.9]
nfans = [1.2, 1.3, 0.5, 1.0]
input = [toes[0],wlrec[0],nfans[0]]
pred = neural_network(input,weights)
print(pred)


###################### Multi input - multi output with hidden states
ih_wgt = [
   # toes % win # fans
    [0.1, 0.2, -0.1], # hid[0]
    [-0.1,0.1, 0.9], # hid[1]
    [0.1, 0.4, 0.1] ] # hid[2] 

hp_wgt = [
    #hid[0] hid[1] hid[2]
    [0.3, 1.1, -0.3], # hurt?
    [0.1, 0.2, 0.0], # win?
    [0.0, 1.3, 0.1] ] # sad?

weights = [ih_wgt, hp_wgt]

def neural_network(input, weights):
    hid = vect_mat_mul(input,weights[0])
    pred = vect_mat_mul(hid,weights[1])
    return pred

toes = [8.5, 9.5, 9.9, 9.0]
wlrec = [0.65,0.8, 0.8, 0.9]
nfans = [1.2, 1.3, 0.5, 1.0]

input = [toes[0],wlrec[0],nfans[0]]
pred = neural_network(input,weights)
print(pred)


###################### Multi input - multi output with hidden states [in numpy]
import numpy as np


# toes % win # fans
ih_wgt = np.array(
    [ [0.1, 0.2, -0.1], # hid[0]
     [-0.1,0.1, 0.9], # hid[1]
     [0.1, 0.4, 0.1]]).T # hid[2]

# hid[0] hid[1] hid[2]
hp_wgt = np.array([ 
    [0.3, 1.1, -0.3], # hurt?
    [0.1, 0.2, 0.0], # win?
    [0.0, 1.3, 0.1] ]).T # sad?

weights = [ih_wgt, hp_wgt]

def neural_network(input, weights):
    hid = input.dot(weights[0])
    pred = hid.dot(weights[1])
    return pred

toes = np.array([8.5, 9.5, 9.9, 9.0])
wlrec = np.array([0.65,0.8, 0.8, 0.9])
nfans = np.array([1.2, 1.3, 0.5, 1.0])

input = np.array([toes[0],wlrec[0],nfans[0]])
pred = neural_network(input,weights)
print(pred)
