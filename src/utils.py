import numpy as np
import pandas as pd
'''Pipeline and helper functions'''

# --- DATA PREPARATION ---

""" Returns list of n by n elements of iterable. """
def grouped(iterable, n):
    return zip(*[iter(iterable)]*n)

""" Transform initial string data into list of 3D body points. """
def str2float(s):
    s = s.replace('[', '')
    s = s.replace(']', '')
    list = s.split(', ')
    list = [float(i) for i in list]

    new_list = np.zeros((17, 3))
    for (point, i) in zip(grouped(list, 4), range(17)):
        #We don't need the CI
        x = point[0]
        y = point[1]
        z = point[2]
        new_list[i] = np.array([x, y, z])
    return new_list

def get_df_points(data, col = 'kp3ds'):
    nbody_points = len(data[col][0])
    col_names = [f"point_{str(i)}" for i in range(nbody_points)]
    my_dict = {k: [] for k in col_names}

    for row in data[col]: 
        for (value, key) in zip(row, my_dict):
            my_dict[key].append(value)
    
    return pd.DataFrame(my_dict)


# --- INTERMEDIARY FEATURES ---

""" Returns the unit vector of the vector.  """
def unit_vector(vector):
    return vector / np.linalg.norm(vector)

""" Returns the angle in radians between vectors 'v1' and 'v2'. 
Vector v1 is point_a - point_b, and vector v2 is point_c - point_b.
Point b is the joint of interest """
def angle(a, b, c):
    v1 = unit_vector(a-b)
    v2 = unit_vector(c-b)

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
    
