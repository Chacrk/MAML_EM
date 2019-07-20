import numpy as np

def func(x):
    result = 0.001
    for i in range(28):
        result = result * x
    return result

print(func(0.999))
print(func(0.99))
print(func(0.95))
print(func(0.9))