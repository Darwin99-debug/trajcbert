import transformers
#import torch.nn as nn
import numpy as np



# sleep for 10 seconds
import time
time.sleep(10)
# create a random array with 1000 elements
x = np.random.rand(1000)
# compute the mean of the array
y = np.mean(x)
# print the result
print(y)

# sleep for 10 seconds
time.sleep(10)

# create a file called 'result.txt' and write the result in it
with open('result.txt', 'w') as fp:
    fp.write(str(y))



# import data.json file and convert it to a dictionary
import json
with open('data.json') as json_file:
    data = json.load(json_file)

# print the dictionary
print(data)
