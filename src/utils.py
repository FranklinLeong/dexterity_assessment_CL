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

# --- HELPER FUNCTIONS ---

"""Returns the closest index for the wanted time period """
def time_id(data, t1, t2, col_name = 'time'):
    closest_index_1 = (data[col_name] - t1).abs().idxmin()
    closest_index_2 = (data[col_name] - t2).abs().idxmin()
    return [closest_index_1, closest_index_2]

# --- INTERMEDIARY FEATURES ---

""" Returns the unit vector of the vector.  """
def unit_vector(vector):
    return vector / np.linalg.norm(vector)

""" Returns the angle in degrees between vectors 'v1' and 'v2'. 
Vector v1 is point_a - point_b, and vector v2 is point_c - point_b.
Point b is the joint of interest """
def angle(a, b, c):
    v1 = unit_vector(a-b)
    v2 = unit_vector(c-b)

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)) * 180 / np.pi
"""
def joint_angles(data, t, joint = 'elbow_r', mvt = 'flexion'):
    if joint == 'elbow_r':
        a = data['shoulder_r'][t]
        b = data[joint][t]
        c = data['wrist_r'][t]
        return angle(a, b, c)
    
    elif joint == 'elbow_l':
        a = data['shoulder_l'][t]
        b = data[joint][t]
        c = data['wrist_l'][t]
        return angle(a, b, c)   
"""

""" Returns mean angular velocity of angular velocities at each timestep
data = angles of joint of interest, t1 and t2 are expressed in seconds. """
def angle_velocity(data, t1, t2, name = 'leftelbow', axis = 0):

    idx = time_id(data, t1, t2)
    angles = [point[axis] for point in data[name][idx[0]:idx[1]]]
    times = data['time'][idx[0]:idx[1]]
    sum = 0
    n = len(angles)

    for i in range(n-1):
        w_i = np.abs((angles[i+1] - angles[i])/(times[i+1 + idx[0]] - times[i + idx[0]]))
        sum = sum + w_i
    
    velocity = 1/n * sum

    return velocity

""" Returns the distance of the total trajectory of one joint given a period of time"""
def dist_trajectory(data, t1, t2, joint_name):
    idx = time_id(data, t1, t2)
    points = data[joint_name][idx[0]:idx[1]]
    distance = 0
    dist_trajectory = 0
    n = len(points)

    for i in range(n-1):
        a = points[i + idx[0]]
        b = points[i+1 + idx[0]]
        distance = np.linalg.norm(b-a)
        dist_trajectory = dist_trajectory + distance

    return dist_trajectory