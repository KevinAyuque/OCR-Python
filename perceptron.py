import numpy as np
from PIL import Image

#im = Image.open("letter_d.png").convert("L")
im = Image.open("letter_d.png").convert("L")
ar = np.array(im)
arr = np.array(ar, dtype=int)

arr[arr == 0] = 1
arr[arr == 255] = -1
print(arr)
print(ar.dtype)
print(arr.dtype)
print(arr.flatten())

threshold = 0

input_n = 63
output_n = 7

b = np.zeros(output_n)
w = np.zeros((input_n, output_n))

t = np.zeros((output_n, output_n))
t.fill(-1)

for i in range(0, output_n - 1):
    t[i, i] = 1

def train( input, output ):
    x = input
    
    
    print "Training starts"
    stopping_condition = False
    while(stopping_condition == False):
        stopping_condition = True
        for i in range(0, input_n - 1):
            for j in range(0, output_n - 1):
                y_in = np.zeros(output_n)
                #print(x)
                #print(w[:,j])
                y_in[j] = b[j] + np.dot(x, w[:,j])
                y = activation(y_in[j], threshold)
        
                if t[output][j] != y:
                    b[j] = b[j] + t[output][j]
                    w[i][j] = w[i][j] + t[output][j]*x[i]
                    stopping_condition = False
        print "Epoch"
   
    print(b)
    print(w)
    
    print "Training complete"

def activation(y_in, threshold):
    if y_in > threshold:
        return 1
    elif -threshold <= y_in and y_in <= threshold:
        return 0
    elif y_in < threshold:
        return -1


#train(np.ones(63),0)