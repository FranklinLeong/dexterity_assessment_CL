import numpy as np
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

    new_list = []
    for x, y, z, ci in grouped(list, 4):
        #We don't need the CI
        new_list.append([x, y, z])
    return new_list

# --- INTERMEDIARY FEATURES ---

""" Returns the unit vector of the vector.  """
def unit_vector(vector):
    return vector / np.linalg.norm(vector)

""" Returns the angle in radians between vectors 'v1' and 'v2'. 
Vector v1 is point_a - point_b, and vector v2 is point_c - point_b. """
#Point B is the middle point
def angle(a, b, c):
    v1 = unit_vector(a-b)
    v2 = unit_vector(c-b)

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
    
