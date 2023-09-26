import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from skspatial.objects import Plane, Points

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

"""Returns the closest index for the wanted time period t1 and t2 in seconds """
def time_id(data, t1, t2, col_name = 'time'):
    closest_index_1 = (data[col_name] - t1).abs().idxmin()
    closest_index_2 = (data[col_name] - t2).abs().idxmin()
    return [closest_index_1, closest_index_2]

"""Gram-Schmidt orthogonalization algorithm, basis vectors are given as rows of vectors"""

def gram_schmidt(vectors):
    num_vectors, vector_length = vectors.shape
    orthogonalized = np.zeros((num_vectors, vector_length))
    for i in range(num_vectors):
        orthogonalized[i] = vectors[i]
        for j in range(i):
            orthogonalized[i] -= np.dot(vectors[i], orthogonalized[j]) / np.dot(orthogonalized[j], orthogonalized[j]) * orthogonalized[j]
        orthogonalized[i] /= np.linalg.norm(orthogonalized[i])
    return orthogonalized

"""Transform original coordinates to new coordinate system based on basis_vectors and center"""
def transform_to_new_coordinates(original_coordinates, basis_vectors, center):
    translated_coords = original_coordinates - center
    transformation_matrix = np.column_stack(basis_vectors) 
    new_coordinates = np.dot(transformation_matrix, translated_coords)
    return new_coordinates

"""Transform original coordinates to body frame coordinates"""
def transform_points(data):
    #Find the projected raw middle feet to the fitted plane of all feet data points 
    feet_points = np.concatenate((data['leftfoot'][::3], data['rightfoot'][::3]))
    feet_points = [a for a in feet_points]

    points = Points(feet_points)
    plane = Plane.best_fit(points)
    
    middle_feet = (data['leftfoot'] + data['rightfoot'])/2
    middle_feet = np.array([plane.project_point(a) for a in middle_feet])
    leftfoot = np.array([np.array(a) for a in data['leftfoot']])
    hips = np.array([np.array(a) for a in data['hips']])

    X = leftfoot - middle_feet 
    Z = hips - middle_feet
    X = [plane.project_vector(x) for x in X]
    C = middle_feet

    #Axis of coordinate system should be unit vector
    X = [np.array(x) for x in X]
    Z = [np.array(z) for z in Z]
    Y = np.cross(X, Z)
    
    #Orthogonalization of these 3 axises using Gram-Schmidt process
    A = [np.array([x, y, z]) for x, y, z in zip(X, Y, Z)]
    #A = [np.array([x, y, Z]) for x, y in zip(X, Y)]
    R = [gram_schmidt(a) for a in A]
    
    #body joints transformed to body frame coordinates
    new_df = pd.DataFrame()
    keys = ['nose', 'lefteye', 'righteye', 'leftear', 'rightear',
            'leftshoulder', 'rightshoulder', 'leftelbow', 'rightelbow',
            'leftwrist', 'rightwrist', 'lefthip', 'righthip', 'leftknee',
            'rightknee', 'leftfoot', 'rightfoot', 'neck', 'hips']
    for key in keys:
        new_df[key] = [transform_to_new_coordinates(point, r, c) for point, r, c in zip(data[key], R, C)]
    new_df['time'] = data['time']
    return new_df


# --- MAKE VIDEOS FUNCTIONS ---

""" Plot one bone between 2 joints"""
def plot_bone(ax, data, joint1, joint2, time, color):
    ax.plot(xs = [data[joint1][time][0], data[joint2][time][0]], 
            ys = [data[joint1][time][1], data[joint2][time][1]],
            zs = [data[joint1][time][2], data[joint2][time][2]], c = color)
    

""" Plot whole skeleton (without head) on a specific time and save png image in plot_img folder"""
def plot_skeleton(data, time, fig, ax):

    ax.set_xlim(-3, 3)
    ax.set_ylim(-4, 4)
    ax.set_zlim(-2, 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(azim=45)
    #ax.view_init(vertical_axis='x', roll = -180, azim = 45)

    cmap = plt.cm.get_cmap('cool')
    colors = []
    for i in range(12): 
        colors.append(cmap(i*25))
        
    bones = [['leftelbow', 'leftwrist'],
             ['lefthip', 'leftknee'],
             ['rightfoot', 'rightknee'],
             ['righthip', 'rightknee'],
             ['lefthip', 'righthip'],
             ['rightshoulder', 'righthip'],
             ['lefthip', 'leftshoulder'],
             ['rightshoulder', 'leftshoulder'],
             ['rightshoulder', 'rightelbow'],
             ['rightwrist', 'rightelbow'],
             ['leftelbow', 'leftshoulder'],
             ['leftfoot', 'leftknee']]
    
    for i in range(12):
        plot_bone(ax, data, bones[i][0], bones[i][1], time, cmap(i*25))

    fig.tight_layout()
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.cla()
    return img

""" Create video from plot_img folder to file_name video. t1 and t2 are in seconds."""
def video_skeleton(file_name, data, t1, t2):    
    t1, t2 = time_id(data, t1, t2)
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(6,6))
    
    fps = 30.0
    size = (600,600)
    out = cv2.VideoWriter(f'videos/{file_name}.avi',cv2.VideoWriter_fourcc(*'MJPG'), fps, size)
    for t in range(t1, t2+1):
        img = plot_skeleton(data, t, fig, ax)
        out.write(img)
    out.release()

# --- PRIMARY FEATURES ---

""" Returns the unit vector of the vector.  """
def unit_vector(vector):
    return vector / np.linalg.norm(vector)

""" Add naive joint angles to dataframe. """
def add_naive_joint_angles(data, parent_joint, joint, child_joint):
    v1 = data[parent_joint] - data[joint]
    v2 = data[child_joint] - data[joint]
    angles = [np.dot(unit_vector(a), unit_vector(b)) for a, b in zip(v1, v2)]
    data[joint+'_angle'] = np.arccos(angles)*180/np.pi

"""Attempt to return angle of a joint projected on a body plane"""
def angle_plane(data, jointA=None, jointB=None, plane='sagittal', trunk = False):
    if jointA and jointB:
        D = data[jointB] - data[jointA]
        D = [unit_vector(d) for d in D]
    
    if plane == 'sagittal':
        N = np.array([0, 0, 1])
        
        if trunk :
            A = data['rightshoulder'] - data['hips']
            B = data['leftshoulder'] - data['hips']
            D = [np.cross(a, b) for a, b in zip(A, B)]
            D = [unit_vector(d) for d in D]
            angle = [np.dot(d, N) for d in D]
            angle = 90 - np.arccos(angle)*180/np.pi

        else:
            D = [np.array([0, d[1], d[2]]) for d in D]
            angle = [np.dot(d, N) for d in D]
            angle = np.arccos(angle)*180/np.pi

            # To define if it is flexion or extension
            for i in range(len(D)):
                norm_front = np.linalg.norm(D[i] - np.array([0, -1, 0]))
                norm_behind = np.linalg.norm(D[i] - np.array([0, 1, 0]))

                if norm_behind < norm_front:
                    angle[i] = -angle[i]

    elif plane == 'transverse':
        N = np.array([-1, 0, 0])
        D = [np.array([d[0], d[1], 0]) for d in D]
        angle = [np.dot(d, N) for d in D]
        angle = np.arccos(angle)*180/np.pi - 90

        for i in range(len(D)):
            if 'left' in jointA:
                angle[i] = -angle[i]

    elif plane =='coronal':
        if trunk:
            A = data['righthip'] - data['hips']
            B = data['neck'] - data['hips']

            A_u = [unit_vector(a) for a in A]
            B_u = [unit_vector(b) for b in B]
            DOT = [np.dot(a_u, b_u) for a_u, b_u in zip(A_u, B_u)]
            angle = np.arccos(DOT)*180/np.pi - 90
        else : 
            N = np.array([0, 0, 1])
            D = [np.array([d[0], 0, d[2]]) for d in D]
            angle = [np.dot(d, N) for d in D]
            #angle = [np.arctan2(np.cross(N, d), np.dot(N, d)) for d in D]
            angle = np.arccos(angle)*180/np.pi 
            # To define if it is on one side or the other
            for i in range(len(D)):
                norm_left = np.linalg.norm(D[i] - np.array([-2, 0, 0]))
                norm_right = np.linalg.norm(D[i] - np.array([2, 0, 0]))
                
                if 'left' in jointA and norm_right < norm_left and not (angle[i] > 150):
                    angle[i] = -angle[i]

                if 'right' in jointA and norm_right > norm_left and not (angle[i] > 150):
                    angle[i] = -angle[i]

    return angle

"""Add positions and angular velocities and accelerations in data"""
def add_velocities_acceleration(data, keys, joints) : 
    times = data['time']
    n = len(data) - 1
    for key in keys:
        #First add the velocities of the body points
        velocities = [0]
        V_x = [0]
        V_y = [0]
        V_z = [0]
        for i in range(n):
            dt = (times[i+1] - times[i])
            a = data[key][i]
            b = data[key][i+1]
            dist = np.linalg.norm(b-a)
            w = dist/dt
            v_x, v_y, v_z = (b-a)/dt
            velocities.append(w)
            V_x.append(v_x)
            V_y.append(v_y)
            V_z.append(v_z)
        data[key +'_V'] = velocities
        add_filtered(data, key+'_V')
        data[key +'_Vx'] = V_x
        data[key +'_Vy'] = V_y
        data[key +'_Vz'] = V_z

        #Then add the accelerations of the body points
        accelerations = [0,0]
        A_x = [0,0]
        A_y = [0,0]
        A_z = [0,0]        
        for i in range(n-1):
            dt = times[i+1] - times[i]
            a_i = (velocities[i+1] - velocities[i])/dt
            a_x = (V_x[i+1] - V_x[i])/dt
            a_y = (V_y[i+1] - V_y[i])/dt
            a_z = (V_z[i+1] - V_z[i])/dt
            accelerations.append(a_i)
            A_x.append(a_x)
            A_y.append(a_y)
            A_z.append(a_z)
        data[key+'_A'] = accelerations
        add_filtered(data, key+'_A')
        data[key+'_Ax'] = A_x
        data[key+'_Ay'] = A_y
        data[key+'_Az'] = A_z

    for joint in joints:
        #Add the angular velocities and its filtered version
        velocities = [0]
        for i in range(n):
            a = data[joint][i]
            b = data[joint][i+1]
            dist = np.linalg.norm(b-a)
            w = dist/(times[i+1] - times[i])
            velocities.append(w)
        data[joint +'_V'] = velocities
        add_filtered(data, joint+'_V')

        #Add the angular accelerations and its filtered version
        accelerations = [0,0]
        for i in range(n-1):
            a_i = (velocities[i+1] - velocities[i])/(times[i+1] - times[i])
            accelerations.append(a_i)

        data[joint+'_A'] = accelerations
        add_filtered(data, joint+'_A')


""" Add centroid of set of joints to data"""
def add_centroid(data, joints, name):
    sum = 0
    for joint in joints:
        sum = sum + data[joint]

    centroid = sum/len(joints)
    data[name] = centroid

"""Filter column of data, denominated by keys, using simple moving average"""
def add_filtered(data, key, window_size = 7):
    data[key+'_filtered'] = data[key].rolling(window_size).mean().fillna(0)

"""Add to dataset the 3 components of body points separately as new variables"""
def add_each_components(data, keys):
    for key in keys:
        data[key+'_x'] = [x for x, y, z in data[key]]
        data[key+'_y'] = [y for x, y, z in data[key]]
        data[key+'_z'] = [z for x, y, z in data[key]]