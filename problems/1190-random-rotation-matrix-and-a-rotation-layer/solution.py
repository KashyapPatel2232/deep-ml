import math
import numpy as np
def rotation_layer(X, angle):
    ident = np.array([[math.cos(angle), -1*math.sin(angle)],[math.sin(angle), math.cos(angle)]])
    X = np.array(X)
    return (X @ ident.T).tolist()
