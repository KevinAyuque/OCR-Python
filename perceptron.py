import numpy as np
import numpy
from PIL import Image
from numpy import genfromtxt

training_n = 5
image_width=7
image_height = 9

# Cantidad de input units
input_n = image_width * image_height

# Cantidad de output units 
output_n = 7

threshold = 0
b = np.zeros(output_n)
w = np.zeros((input_n, output_n))
t = np.zeros((output_n, output_n))
t.fill(-1)

for i in range(0, output_n):
    t[i, i] = 1
    
def blockshaped(arr, nrows, ncols):
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))
#Creado por unutbu de Stack Overflow

def imageToArray(image):
    image_array = np.array(image, dtype=int)
    image_array[image_array < 255] = 1
    image_array[image_array == 255] = -1
    return image_array
    
def activation(y_in, threshold):
    if y_in > threshold:
        return 1
    elif -threshold <= y_in and y_in <= threshold:
        return 0
    elif y_in < threshold:
        return -1
        
def interpretResult(result):
    for i in range(0, result.size):
        if result[i] == 1:
            if i == 0:
                print "Puede ser A"
            elif i == 1:
                print "Puede ser B"
            elif i == 2:
                print "Puede ser C"
            elif i == 3:
                print "Puede ser D"
            elif i == 4:
                print "Puede ser E"
            elif i == 5:
                print "Puede ser J"
            else:
                print "Puede ser K"

def train( input, output ):
    x = input

    print "Training starts"
    stopping_condition = False
    while(stopping_condition == False):
        stopping_condition = True
        for i in range(0, input_n):
            y_in = np.zeros(output_n)
            y = np.zeros(output_n)
            
            for j in range(0, output_n):
                y_in[j] = b[j] + np.dot(x, w[:,j])
                y[j] = activation(y_in[j], threshold)
                
            for j in range(0, output_n):
                if t[output][j] != y[j]:
                    b[j] = b[j] + t[output][j]
                    for i2 in range(0, input_n):
                        old_w = w[i2][j]
                        w[i2][j] = w[i2][j] + t[output][j]*x[i2]
                    
                        if old_w != w[i2][j]:
                            stopping_condition = False
        print "Epoch"
    print "Training complete"

def classify(input):
    x = input
    y_in = np.zeros(output_n)
    y = np.zeros(output_n)
    for j in range(0, output_n):
        y_in[j] = b[j] + np.dot(x, w[:,j])
        y[j] = activation(y_in[j], threshold)
    return y
    
training_data_image = Image.open("input.png").convert("L")
training_data_array = imageToArray(training_data_image)
training_data_array = blockshaped(training_data_array, image_height, image_width)

output_goal = 0
for input in training_data_array:
    train(input.flatten(), output_goal)
    output_goal += 1
    
    if output_goal == 7:
        output_goal = 0
        
character_image = Image.open("test/j_1.png").convert("L")
#character_image.show()
character_array = imageToArray(character_image)
character_result = classify(character_array.flatten())
#print(character_array)
#print(character_result)
interpretResult(character_result)