import numpy as np

x = np.array([1, 2, 3])

def hello():
    if __name__ == '__main__':
        print('hello from main')
    else:
        print('hello')
        print(__name__)

hello()