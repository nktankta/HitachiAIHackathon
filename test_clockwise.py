import numpy as np

unclock = np.array([
    [[1,0],[2,1],[1,2],[0,1]],
    [[0,0],[1,0],[1,1],[0,1]],
    [[2,0],[10,8],[8,9],[0,1]]
    ])

clock = np.array([
    [[1,0],[0,1],[1,2],[2,1],],
    [[0,0],[0,1],[1,1],[1,0]],
    [[2,0],[0,1],[8,9],[10,8]]
    ])

def is_clockwise(arr):
    diagonal = arr[2]-arr[0]
    phi = np.arctan(diagonal[1]/(diagonal[0]+1e-5)) + (diagonal[0]<0) * np.pi
    middle = arr[1]-arr[0]
    phi2 = np.arctan(middle[1]/(middle[0]+1e-5))  + (middle[0]<0) * np.pi
    return (phi-phi2)% (2*np.pi) > np.pi

if __name__ == '__main__':
    for i in unclock:
        print(is_clockwise(i))
    for i in clock:
        print(is_clockwise(i))