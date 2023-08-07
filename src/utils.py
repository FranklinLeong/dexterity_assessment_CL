import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import cv2
import os

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


# --- MAKE VIDEOS FUNCTIONS ---

""" Plot one bone between 2 joints"""
def plot_bone(ax, data, joint1, joint2, time, color):
    ax.plot(xs = [data[joint1][time][0], data[joint2][time][0]], 
            ys = [data[joint1][time][1], data[joint2][time][1]],
            zs = [data[joint1][time][2], data[joint2][time][2]], c = color)
    

""" Plot whole skeleton (without head) on a specific time and save png image in plot_img folder"""
def plot_skeleton(data, time, fig, ax):

    ax.set_xlim3d(2, 4)
    ax.set_ylim3d(-2, 3)
    ax.set_zlim3d(-1.5, 0.5)
    ax.view_init(azim=90)
    """ ax.set_xlim(-2, 1)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 1) """
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #ax.view_init(vertical_axis='x', roll = -180, azim = 45)

    cmap = plt.cm.get_cmap('cool')
    colors = []
    for i in range(12): 
        colors.append(cmap(i*25))
        
    plot_bone(ax, data, 'leftfoot', 'leftknee', time, colors[0])
    plot_bone(ax, data, 'lefthip', 'leftknee', time, colors[1])
    plot_bone(ax, data, 'rightfoot', 'rightknee', time, colors[2])
    plot_bone(ax, data, 'righthip', 'rightknee', time, colors[3])
    plot_bone(ax, data, 'lefthip', 'righthip', time, colors[4])
    plot_bone(ax, data, 'rightshoulder', 'righthip', time, colors[5])
    plot_bone(ax, data, 'lefthip', 'leftshoulder', time, colors[6])
    plot_bone(ax, data, 'rightshoulder', 'leftshoulder', time, colors[7])
    plot_bone(ax, data, 'rightshoulder', 'rightelbow', time, colors[8])
    plot_bone(ax, data, 'rightwrist', 'rightelbow', time, colors[9])
    plot_bone(ax, data, 'leftelbow', 'leftshoulder', time, colors[10])
    plot_bone(ax, data, 'leftelbow', 'leftwrist', time, colors[10])

    fig.tight_layout()
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    """     output_path = f'plot_img/kde_{time}.png'
    plt.savefig(output_path)
    plt.cla() # needed to remove the plot because savefig doesn't clear it """
    plt.cla()
    return img

""" Create img plots on a period of time"""
""" def skeleton_frames(data, t1, t2):
    images = []
    for t in range(t1, t2+1):
        img = plot_skeleton(data, t)
        images.append(img)

    return images """

""" Create video from plot_img folder to file_name video"""
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

""" Empty plot_img folder"""
def empty_plot_img():
    removing_files = glob.glob('plot_img/*.png')
    for i in removing_files:
        os.remove(i)


# --- PLOT FUNCTIONS ---

""" Plot joint angles around 3 axis during a period of an event"""
""" def plot_angles(data, t1, t2, joint_name, event, x_vertical1 = None, x_vertical2 = None, label_vertical1=None, label_vertical2=None, vertical_line1 = False, vertical_line2 = False):
    idx = time_id(data, t1, t2)
    angles_x = [point[0] for point in data[joint_name][idx[0]:idx[1]]]
    angles_y = [point[1] for point in data[joint_name][idx[0]:idx[1]]]
    angles_z = [point[2] for point in data[joint_name][idx[0]:idx[1]]]
    time = data['time'][idx[0]:idx[1]]
    
    plt.figure(figsize=(12,5))
    if 'shoulder' in joint_name:
        plt.plot(time, angles_x, label = 'Rotation around x-axis (flexion)')
        plt.plot(time, angles_y, label = 'Rotation around y-axis')
        plt.plot(time, angles_z, label = 'Rotation around z-axis (abduction)')
    elif 'elbow' in joint_name:
        plt.plot(time, angles_x, label = 'Rotation around x-axis')
        plt.plot(time, angles_y, label = 'Rotation around y-axis')
        plt.plot(time, angles_z, label = 'Rotation around z-axis (flexion)')
    if 'trunk' in joint_name:
        plt.plot(time, angles_x, label = 'Rotation around x-axis (flexion)')
        plt.plot(time, angles_y, label = 'Rotation around y-axis')
        plt.plot(time, angles_z, label = 'Rotation around z-axis (lateral flexion)')
    if 'hip' in joint_name:
        plt.plot(time, angles_x, label = 'Rotation around x-axis (flexion)')
        plt.plot(time, angles_y, label = 'Rotation around y-axis')
        plt.plot(time, angles_z, label = 'Rotation around z-axis (lateral flexion)')
    if 'neck' in joint_name:
        plt.plot(time, angles_x, label = 'Rotation around x-axis (forward flexion)')
        plt.plot(time, angles_y, label = 'Rotation around y-axis')
        plt.plot(time, angles_z, label = 'Rotation around z-axis (lateral flexion)')

    if vertical_line1:
        plt.axvline(x_vertical1, label = label_vertical1, linestyle = '--', color = 'grey')

    if vertical_line2:
        plt.axvline(x_vertical2, label = label_vertical2, linestyle = '--', color = 'red')

    plt.gca().invert_yaxis()
    plt.title(f'{joint_name} when {event}')
    plt.xlabel('Time [s]')
    plt.ylabel('Angle [deg]')
    plt.legend()
    plt.show() """

""" Plots all average joint velocities given a period of time in an event. """
def plot_joint_velocities(data, t1, t2, event, with_error = False):
    keys = [joint for joint in data.columns.values if 'angles' in joint]
    velocities_x = np.array([angle_velocity(data, t1, t2, name = key, axis = 0) for key in keys])
    velocities_y = np.array([angle_velocity(data, t1, t2, name = key, axis = 1) for key in keys])
    velocities_z = np.array([angle_velocity(data, t1, t2, name = key, axis = 2) for key in keys])
    plt.figure(figsize=(15,6))

    if with_error:
        plt.errorbar(x = keys, y = velocities_x[:,0], yerr = velocities_x[:,1], label = 'Rotation around x-axis', capsize = 5, fmt='o')
        plt.errorbar(x = keys, y = velocities_y[:,0], yerr = velocities_y[:,1], label = 'Rotation around y-axis', capsize=5, fmt = 'x')
        plt.errorbar(x = keys, y = velocities_z[:,0], yerr = velocities_z[:,1], label = 'Rotation around z-axis', capsize=5, fmt = 'v')
    else:
        plt.scatter(x = keys, y = velocities_x[:,0], label = 'Rotation around x-axis')
        plt.scatter(x = keys, y = velocities_y[:,0], label = 'Rotation around y-axis')
        plt.scatter(x = keys, y = velocities_z[:,0], label = 'Rotation around z-axis')

    plt.title(f'Average joint velocities when {event}')
    plt.ylabel('Average joint velocities [deg/s]')
    plt.legend()
    plt.show()

"""Plot one joint during a period of time"""
def plot_joint(df, t1, t2, joint_l, joint_r, title, trunk = False):
    x = df['time'][t1:t2]
    if trunk == True:
        plt.plot(x, df[joint_l][t1:t2])
    else:
        plt.plot(x, df[joint_l][t1:t2], label = 'Left')
        plt.plot(x, df[joint_r][t1:t2], label = 'Right')
        plt.legend()
    plt.ylabel('Degree [°]')
    plt.xlabel('Time [s]')
    plt.title(title)

"""Plot all joints during a period of time"""
def plot_all_joints(data, t1, t2, title):
    t1, t2 = time_id(data, t1, t2)
    fig = plt.figure(figsize=(12, 6))
    fig.suptitle(title)

    plt.subplot(2, 3, 1)
    plot_joint(data, t1, t2, 'leftelbow_flex', 'rightelbow_flex', 'Elbow flexion')

    plt.subplot(2, 3, 2)
    plot_joint(data, t1, t2, 'leftshoulder_flex', 'rightshoulder_flex', 'Shoulder flexion')

    plt.subplot(2, 3, 3)
    plot_joint(data, t1, t2, 'leftshoulder_abduc', 'rightshoulder_abduc', 'Shoulder abduction')

    plt.subplot(2, 3, 4)
    plot_joint(data, t1, t2, 'trunk_forward_flex', '', title = 'Trunk forward flexion', trunk=True)

    plt.subplot(2, 3, 5)
    plot_joint(data, t1, t2, 'trunk_lateral_flex', '', title = 'Trunk lateral flexion', trunk=True)

    plt.subplot(2, 3, 6)
    plot_joint(data, t1, t2, 'trunk_rotation', '', title = 'Trunk transverse rotation', trunk=True)

    plt.tight_layout()

# --- INTERMEDIARY FEATURES ---

""" Returns the unit vector of the vector.  """
def unit_vector(vector):
    return vector / np.linalg.norm(vector)

""" Add naive joint angles to dataframe. """
def add_naive_joint_angles(data, parent_joint, joint, child_joint):
    v1 = data[parent_joint] - data[joint]
    v2 = data[child_joint] - data[joint]
    angles = []
    for v1_u, v2_u in zip(v1, v2):
        v1_u = unit_vector(v1_u)
        v2_u = unit_vector(v2_u)
        angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) * 180 / np.pi
        angles.append(angle)
    data[joint+'_angle'] = angles

""" Returns mean angular velocity of angular velocities at each timestep
data = angles of joint of interest, t1 and t2 are expressed in seconds. """
def angle_velocity(data, t1, t2, name = 'leftelbow_angles', axis = 0):

    idx = time_id(data, t1, t2)
    angles = [point[axis] for point in data[name][idx[0]:idx[1]]]
    times = data['time'][idx[0]:idx[1]]
    sum = 0
    n = len(angles)
    w = []

    for i in range(n-1):
        w_i = np.abs((angles[i+1] - angles[i])/(times[i+1 + idx[0]] - times[i + idx[0]]))
        w.append(w_i)
    
    velocity = 1/n * np.sum(w)
    std = np.std(w)

    return (velocity, std)

""" Returns mean angular velocity of angular velocities at each timestep
data = angles of joint of interest, t1 and t2 are expressed in seconds. """
def angle_velocity2(data, t1, t2, name = 'leftelbow_angles'):

    idx = time_id(data, t1, t2)
    angles = data[name][idx[0]:idx[1]]
    times = data['time'][idx[0]:idx[1]]
    sum = 0
    n = len(angles)
    w = []

    for i in range(n-1):
        w_i = np.abs((angles[i+1 + idx[0]] - angles[i + idx[0]])/(times[i+1 + idx[0]] - times[i + idx[0]]))
        w.append(w_i)
    
    velocity = 1/n * np.sum(w)
    std = np.std(w)

    return (velocity, std)

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

""" Returns reach distance of left or right wrist"""
def reach_dist(data, t1, t2, left=False, right=False):
    t1, t2 = time_id(data, t1, t2)
    if left:
        points = data['leftwrist'][t1:t2]
    if right:
        points = data['rightwrist'][t1:t2]

    dist_trajectory = 0
    n = len(points)
    for i in range(n-1):
        a = points[i + t1]
        b = points[i+1 + t1]
        distance = np.linalg.norm(b-a)
        dist_trajectory = dist_trajectory + distance    
    
    return dist_trajectory
    


"""Returns the exact time.. ?"""
def time_trajectory(data, t1, t2): 
    idx = time_id(data, t1, t2)
    time = data['time'][idx[0]] - data['time'][idx[1]]

    return np.abs(time)

""" Returns the straight line distance between 2 points"""
def distance_AB(data, t1, t2, joint_name):
    idx = time_id(data, t1, t2)
    point_a = data[joint_name][idx[0]]
    point_b = data[joint_name][idx[1]]
    distance = np.linalg.norm(point_b - point_a)

    return distance

# --- FINAL FEATURES ---

""" Return the Range of Motion. """
def RoM(data, t1, t2, joint, axis = None): 
    idx = time_id(data, t1, t2)
    
    if axis == None:
        angles = data[joint][idx[0]:idx[1]]
    else: 
        angles = [value[axis] for value in data[joint][idx[0]:idx[1]]]
        
    max_angle = np.max(angles)
    min_angle = np.min(angles)
    RoM = max_angle - min_angle

    return RoM

"""Add joint angles DoF in dataframe"""
def add_joint_angles(df):

    # Add middle of the hip as another "joint position"
    difference = (df['lefthip'] - df['righthip'])/2
    #hips = (df['lefthip'] + df['righthip'])/2
    hips = df['righthip'] + difference
    df['hips'] = hips

    # Add middle of the shoulders as another "joint position"
    difference = (df['leftshoulder'] - df['rightshoulder'])/2
    #neck = (df['leftshoulder'] + df['rightshoulder'])/2
    neck = df['rightshoulder'] + difference
    df['neck'] = neck

    # For the left shoulder
    A = df['rightshoulder'] - df['leftshoulder']
    B = df['lefthip'] - df['leftshoulder']
    D = df['leftelbow'] - df['leftshoulder']

    B_u = [unit_vector(b) for b in B]
    D_u = [unit_vector(d) for d in D]
    N = [np.cross(a, b) for a, b in zip(A, B)]

    # d projection
    N2 = [np.cross(n, b) for n, b in zip(N, B)]
    # finding norm of the vector N2 
    N2_norm = [np.linalg.norm(n) for n in N2]
    D_proj = [d - unit_vector((np.dot(d, n)/norm**2)*n) for d, n, norm in zip(D_u, N2, N2_norm)]

    angle = [np.dot(d, b) for d, b in zip(D_proj, B_u)]
    angle = np.arccos(angle)*180/np.pi

    # To define if it is flexion or extension
    N_behind = [np.cross(b, a) for a, b in zip(A, B)]
    for i in range(len(N)):
        norm_front = np.linalg.norm(D_proj[i] - N[i])
        norm_behind = np.linalg.norm(D_proj[i] - (N_behind[i]))

        if norm_behind < norm_front:
            angle[i] = -angle[i]

    df['leftshoulder_flex'] = angle

    # For the right shoulder
    A = df['leftshoulder'] - df['rightshoulder']
    B = df['righthip'] - df['rightshoulder']
    D = df['rightelbow'] - df['rightshoulder']

    B_u = [unit_vector(b) for b in B]
    D_u = [unit_vector(d) for d in D]
    N = [np.cross(b, a) for a, b in zip(A, B)]

    # d projection
    N2 = [np.cross(b, n) for n, b in zip(N, B)]
    # finding norm of the vector N2 
    N2_norm = [np.linalg.norm(n) for n in N2]
    D_proj = [d - unit_vector((np.dot(d, n)/norm**2)*n) for d, n, norm in zip(D_u, N2, N2_norm)]

    angle = [np.dot(d, b) for d, b in zip(D_proj, B_u)]
    angle = np.arccos(angle)*180/np.pi

    # To define if it is flexion or extension
    N_behind = [np.cross(a, b) for a, b in zip(A, B)]
    for i in range(len(N)):
        norm_front = np.linalg.norm(D_proj[i] - N[i])
        norm_behind = np.linalg.norm(D_proj[i] - N_behind[i])

        if norm_behind < norm_front:
            angle[i] = -angle[i]

    df['rightshoulder_flex'] = angle

    A = df['rightshoulder'] - df['leftshoulder']
    B = df['lefthip'] - df['leftshoulder']
    D = df['leftelbow'] - df['leftshoulder']

    A_u = [unit_vector(a) for a in A]
    B_u = [unit_vector(b) for b in B]
    D_u = [unit_vector(d) for d in D]
    N = [np.cross(a, b) for a, b in zip(A, B)]

    # d projection
    N_norm = [np.linalg.norm(n) for n in N]
    D_proj = [d - unit_vector((np.dot(d, n)/norm**2)*n) for d, n, norm in zip(D_u, N, N_norm)]
    angle = [np.dot(d, b) for d, b in zip(D_proj, B_u)]
    angle = np.arccos(angle)*180/np.pi
    """     angle = [np.dot(d, a) for d, a in zip(D_proj, A_u)]
    angle = np.arccos(angle)*180/np.pi - 90

    B_dotted = df['leftshoulder'] - df['lefthip']
    A_dotted = df['leftshoulder'] - df['rightshoulder']

    for i in range(len(N)):
        # To define if the movement is lower or higher than the shoulder, and inner or external of body
        norm_high = np.linalg.norm(D_proj[i] - B_dotted[i])
        norm_low = np.linalg.norm(D_proj[i] - B[i])
        norm_ext = np.linalg.norm(D_proj[i] - A_dotted[i])
        norm_in = np.linalg.norm(D_proj[i] - A[i])

        if norm_in < norm_ext and norm_low < norm_high:
            continue
        elif norm_in > norm_ext and norm_low < norm_high:
            continue
        elif norm_in > norm_ext and norm_low > norm_high:
            angle[i] = 180 - angle[i]
        elif norm_in < norm_ext and norm_low > norm_high:
            angle[i] = angle[i] - 90 """

    df['leftshoulder_abduc'] = angle

    A = df['leftshoulder'] - df['rightshoulder']
    B = df['righthip'] - df['rightshoulder'] 
    D = df['rightelbow'] - df['rightshoulder']

    A_u = [unit_vector(a) for a in A]
    B_u = [unit_vector(b) for b in B]
    D_u = [unit_vector(d) for d in D]
    N = [np.cross(b, a) for a, b in zip(A, B)]

    # d projection
    N_norm = [np.linalg.norm(n) for n in N]
    D_proj = [d - unit_vector((np.dot(d, n)/norm**2)*n) for d, n, norm in zip(D_u, N, N_norm)]
    angle = [np.dot(d, b) for d, b in zip(D_proj, B_u)]
    angle = np.arccos(angle)*180/np.pi
    """ angle = [np.dot(d, a) for d, a in zip(D_proj, A_u)]
    angle = np.arccos(angle)*180/np.pi - 90

    B_dotted = df['righthip'] - df['rightshoulder']
    A_dotted = df['rightshoulder'] - df['leftshoulder']

    for i in range(len(N)):
        # To define if the movement is lower or higher than the shoulder, and inner or external of body
        norm_high = np.linalg.norm(D_proj[i] - B_dotted[i])
        norm_low = np.linalg.norm(D_proj[i] - B[i])
        norm_ext = np.linalg.norm(D_proj[i] - A_dotted[i])
        norm_in = np.linalg.norm(D_proj[i] - A[i])

        if norm_in < norm_ext and norm_low < norm_high:
            continue
        elif norm_in > norm_ext and norm_low < norm_high:
            continue
        elif norm_in > norm_ext and norm_low > norm_high:
            angle[i] = 180 - angle[i]
        elif norm_in < norm_ext and norm_low > norm_high:
            angle[i] = angle[i] - 90 """

    df['rightshoulder_abduc'] = angle

    A = df['leftshoulder'] - df['leftelbow']
    B = df['leftwrist'] - df['leftelbow']

    angle = [np.dot(unit_vector(a), unit_vector(b)) for a, b in zip(A, B)]
    angle = np.arccos(angle)*180/np.pi

    df['leftelbow_flex'] = angle

    A = df['rightelbow'] - df['rightshoulder']
    B = df['rightelbow'] - df['rightwrist']

    angle = [np.dot(unit_vector(a), unit_vector(b)) for a, b in zip(A, B)]
    angle = np.arccos(angle)*180/np.pi

    df['rightelbow_flex'] = angle

    A = df['rightshoulder'] - df['hips']
    B = df['leftshoulder'] - df['hips']
    N = [np.cross(a, b) for a, b in zip(A, B)]
    v_z = np.array([0, 0, 1])

    V1_u = [unit_vector(n) for n in N]
    DOT = [np.dot(v1_u, v_z) for v1_u in V1_u]
    angle = 90 - np.arccos(DOT)*180/np.pi

    df['trunk_forward_flex'] = angle

    A = df['righthip'] - df['hips']
    B = df['neck'] - df['hips']

    A_u = [unit_vector(a) for a in A]
    B_u = [unit_vector(b) for b in B]
    DOT = [np.dot(a_u, b_u) for a_u, b_u in zip(A_u, B_u)]
    angle = np.arccos(DOT)*180/np.pi - 90

    df['trunk_lateral_flex'] = angle

    A = df['righthip'] - df['hips']
    B = df['neck'] - df['hips']
    C = df['rightshoulder'] - df['neck']
    N = [np.cross(b, a) for a, b in zip(A, B)]

    N2 = [np.cross(a, n) for a, n in zip(A, N)]
    N2_norm = [np.linalg.norm(n2) for n2 in N2]

    C_proj = [c - (np.dot(c, n2)/norm2**2)*n2 for c, n2, norm2 in zip(C, N2, N2_norm)]

    A_u = [unit_vector(a) for a in A]
    C_proj = [unit_vector(c) for c in C_proj]
    DOT = [np.dot(a_u, c) for a_u, c in zip(A_u, C_proj)]
    angle = np.arccos(DOT)*180/np.pi

    # To define if it is right or left rotation 
    N_behind = [np.cross(a, b) for a, b in zip(A, B)]
    for i in range(len(N)):
        norm_front = np.linalg.norm(C_proj[i] - N[i])
        norm_behind = np.linalg.norm(C_proj[i] - N_behind[i])

        if norm_behind < norm_front:
            angle[i] = -angle[i]

    df['trunk_rotation'] = angle


""" Add joint velocities of joints of interest to dataframe"""
def add_joint_velocities(df, joints):
    times = df['time']
    n = len(times)
    for joint in joints:
        angles = df[joint]
        w = [0]
        for i in range(n-1):
            w_i = (angles[i+1] - angles[i])/(times[i+1] - times[i])
            w.append(w_i)
        df[joint + '_w'] = w
