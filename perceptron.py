import numpy as np
import numpy
from PIL import Image
from numpy import genfromtxt

threshold = 0.5

training_n = 5
image_width=7
image_height = 9
input_n = image_width * image_height
output_n = 7

#im = Image.open("letter_d.png").convert("L")
im = Image.open("input.png").convert("L")
ar = np.array(im)
total_input = np.array(ar, dtype=int)

total_input[total_input == 0] = 1
total_input[total_input == 255] = -1
#print(total_input)
#print(total_input.reshape((45, 7, 7)))

ex = np.arange(24)
#print(ex)
exx = ex.reshape(6, 4)
#rint(exx)
exxx = exx.reshape(3,2,2,2).swapaxes(1,2)

#newt = total_input.reshape(5,7,9,7).swapaxes(1,3)
def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))
               

my_data = genfromtxt('matriz.csv', delimiter = ',')
newt = blockshaped(total_input, 9, 7)
#print(newt)
#print(newt[2])
#print(exxx.swapaxes(0,2))
#print(exxx)
#print(arr.flatten())



b = np.zeros(output_n)
w = np.zeros((input_n, output_n))

t = np.zeros((output_n, output_n))
t.fill(-1)

for i in range(0, output_n):
    t[i, i] = 1

def train( input, output ):
    x = input
    
    
    print "Training starts"
    stopping_condition = False
    while(stopping_condition == False):
        stopping_condition = True
        for i in range(0, input_n):
            y_in = np.zeros(output_n)
            for j in range(0, output_n):
                
                #print(x)
                #print(w[:,j])
                y_in[j] = b[j] + np.dot(x, w[:,j])
                y = activation(y_in[j], threshold)
        
                if t[output][j] != y:
                    b[j] = b[j] + t[output][j]
                    old_w = w[i][j]
                    w[i][j] = w[i][j] + t[output][j]*x[i]
                    
                    if old_w == w[i][j]:
                        stopping_condition = False
                        print "No weight change"
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

output_goal = 0
for input in newt:
    print(output_goal)
    print(input)
    train(input.flatten(), output_goal)
    output_goal += 1
    
    if output_goal == 7:
        output_goal = 0
np.set_printoptions(threshold=numpy.nan)
print newt