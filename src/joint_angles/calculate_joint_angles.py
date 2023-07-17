import numpy as np
import copy
from scipy.signal import medfilt

# --- Helper functions -- 

def get_R_x(theta):
    R = np.array([[1, 0, 0],
                  [0, np.cos(theta), -np.sin(theta)],
                  [0, np.sin(theta),  np.cos(theta)]])
    return R

def get_R_y(theta):
    R = np.array([[np.cos(theta), 0, np.sin(theta)],
                  [0, 1, 0],
                  [-np.sin(theta), 0,  np.cos(theta)]])
    return R

def get_R_z(theta):
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1]])
    return R


#calculate rotation matrix to take A vector to B vector
def Get_R(A,B):

    #get unit vectors
    uA = A/np.sqrt(np.sum(np.square(A)))
    uB = B/np.sqrt(np.sum(np.square(B)))

    #get products
    dotprod = np.sum(uA * uB)
    crossprod = np.sqrt(np.sum(np.square(np.cross(uA,uB)))) #magnitude

    #get new unit vectors
    u = uA
    v = uB - dotprod*uA
    v = v/np.sqrt(np.sum(np.square(v)))
    w = np.cross(uA, uB)
    w = w/np.sqrt(np.sum(np.square(w)))

    #get change of basis matrix
    C = np.array([u, v, w])

    #get rotation matrix in new basis
    R_uvw = np.array([[dotprod, -crossprod, 0],
                      [crossprod, dotprod, 0],
                      [0, 0, 1]])

    #full rotation matrix
    R = C.T @ R_uvw @ C
    #print(R)
    return R

#Same calculation as above using a different formalism
def Get_R2(A, B):

    #get unit vectors
    uA = A/np.sqrt(np.sum(np.square(A)))
    uB = B/np.sqrt(np.sum(np.square(B)))

    v = np.cross(uA, uB)
    s = np.sqrt(np.sum(np.square(v)))
    c = np.sum(uA * uB)

    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])

    R = np.eye(3) + vx + vx@vx*((1-c)/s**2)

    return R


#decomposes given R matrix into rotation along each axis. In this case Rz @ Ry @ Rx
def Decompose_R_ZYX(R):

    #decomposes as RzRyRx. Note the order: ZYX <- rotation by x first
    thetaz = np.arctan2(R[1,0], R[0,0])
    thetay = np.arctan2(-R[2,0], np.sqrt(R[2,1]**2 + R[2,2]**2))
    thetax = np.arctan2(R[2,1], R[2,2])

    return thetaz, thetay, thetax

def Decompose_R_ZXY(R):

    #decomposes as RzRXRy. Note the order: ZXY <- rotation by y first
    thetaz = np.arctan2(-R[0,1], R[1,1])
    thetay = np.arctan2(-R[2,0], R[2,2])
    thetax = np.arctan2(R[2,1], np.sqrt(R[2,0]**2 + R[2,2]**2))

    return thetaz, thetay, thetax

def read_keypoints(filename):

    num_keypoints = 12
    fin = open(filename, 'r')

    kpts = []
    while(True):
        line = fin.readline()
        if line == '': break

        line = line.split()
        line = [float(s) for s in line]

        line = np.reshape(line, (num_keypoints, -1))
        kpts.append(line)

    kpts = np.array(kpts)
    return kpts


def convert_to_dictionary(kpts):
    #its easier to manipulate keypoints by joint name
    keypoints_to_index = {'lefthip': 6, 'leftknee': 8, 'leftfoot': 10,
                          'righthip': 7, 'rightknee': 9, 'rightfoot': 11,
                          'leftshoulder': 0, 'leftelbow': 2, 'leftwrist': 4,
                          'rightshoulder': 1, 'rightelbow': 3, 'rightwrist': 5}

    kpts_dict = {}
    for key, k_index in keypoints_to_index.items():
        kpts_dict[key] = kpts[:,k_index]

    kpts_dict['joints'] = list(keypoints_to_index.keys())

    return kpts_dict


def add_hips_and_neck(kpts):
    #we add two new keypoints which are the mid point between the hips and mid point between the shoulders

    #add hips kpts
    difference = kpts['lefthip'] - kpts['righthip']
    difference = difference/2
    hips = kpts['righthip'] + difference
    kpts['hips'] = hips
    kpts['joints'].append('hips')


    #add neck kpts
    difference = kpts['leftshoulder'] - kpts['rightshoulder']
    difference = difference/2
    neck = kpts['rightshoulder'] + difference
    kpts['neck'] = neck
    kpts['joints'].append('neck')

    #define the hierarchy of the joints
    hierarchy = {
                 'hips': [],
                 'lefthip': ['hips'], 'leftknee': ['lefthip', 'hips'], 'leftfoot': ['leftknee', 'lefthip', 'hips'],
                 'righthip': ['hips'], 'rightknee': ['righthip', 'hips'], 'rightfoot': ['rightknee', 'righthip', 'hips'],
                 'neck': ['hips'],
                 'leftshoulder': ['neck', 'hips'], 'leftelbow': ['leftshoulder', 'neck', 'hips'], 'leftwrist': ['leftelbow', 'leftshoulder', 'neck', 'hips'],
                 'rightshoulder': ['neck', 'hips'], 'rightelbow': ['rightshoulder', 'neck', 'hips'], 'rightwrist': ['rightelbow', 'rightshoulder', 'neck', 'hips']
                }

    kpts['hierarchy'] = hierarchy
    kpts['root_joint'] = 'hips'

    return kpts

#remove jittery keypoints by applying a median filter along each axis
def median_filter(kpts, window_size = 3):

    filtered = copy.deepcopy(kpts)


    #apply median filter to get rid of poor keypoints estimations
    for joint in filtered['joints']:
        joint_kpts = filtered[joint]
        xs = [value[0] for value in joint_kpts]
        ys = [value[1] for value in joint_kpts]
        zs = [value[2] for value in joint_kpts]
        xs = medfilt(xs, window_size)
        ys = medfilt(ys, window_size)
        zs = medfilt(zs, window_size)
        filtered[joint] = np.stack([xs, ys, zs], axis = -1)

    return filtered

def get_bone_lengths(kpts):

    """
    We have to define an initial skeleton pose(T pose).
    In this case we need to known the length of each bone.
    Here we calculate the length of each bone from data
    """

    bone_lengths = {}
    for joint in kpts['joints']:
        if joint == 'hips': continue
        parent = kpts['hierarchy'][joint][0]

        joint_kpts = kpts[joint]
        parent_kpts = kpts[parent]

        _bone = joint_kpts - parent_kpts
        _bone_lengths = np.sqrt(np.sum(np.square(_bone), axis = -1))

        _bone_length = np.median(_bone_lengths)
        bone_lengths[joint] = _bone_length

        # plt.hist(bone_lengths, bins = 25)
        # plt.title(joint)
        # plt.show()

    #print(bone_lengths)
    kpts['bone_lengths'] = bone_lengths
    return

#Here we define the T pose and we normalize the T pose by the length of the hips to neck distance.
def get_base_skeleton(kpts, normalization_bone = 'neck'):

    #this defines a generic skeleton to which we can apply rotations to
    body_lengths = kpts['bone_lengths']

    #define skeleton offset directions
    offset_directions = {}
    offset_directions['lefthip'] = np.array([1,0,0])
    offset_directions['leftknee'] = np.array([0,-1, 0])
    offset_directions['leftfoot'] = np.array([0,-1, 0])

    offset_directions['righthip'] = np.array([-1,0,0])
    offset_directions['rightknee'] = np.array([0,-1, 0])
    offset_directions['rightfoot'] = np.array([0,-1, 0])

    offset_directions['neck'] = np.array([0,1,0])
    #add ground value

    offset_directions['leftshoulder'] = np.array([1,0,0])
    offset_directions['leftelbow'] = np.array([1,0,0])
    offset_directions['leftwrist'] = np.array([1,0,0])

    offset_directions['rightshoulder'] = np.array([-1,0,0])
    offset_directions['rightelbow'] = np.array([-1,0,0])
    offset_directions['rightwrist'] = np.array([-1,0,0])

    #set bone normalization length. Set to 1 if you dont want normalization
    normalization = kpts['bone_lengths'][normalization_bone]
    #normalization = 1


    #base skeleton set by multiplying offset directions by measured bone lengths. In this case we use the average of two sided limbs. E.g left and right hip averaged
    base_skeleton = {'hips': np.array([0,0,0])}
    def _set_length(joint_type):
        base_skeleton['left' + joint_type] = offset_directions['left' + joint_type] * ((body_lengths['left' + joint_type] + body_lengths['right' + joint_type])/(2 * normalization))
        base_skeleton['right' + joint_type] = offset_directions['right' + joint_type] * ((body_lengths['left' + joint_type] + body_lengths['right' + joint_type])/(2 * normalization))

    _set_length('hip')
    _set_length('knee')
    _set_length('foot')
    _set_length('shoulder')
    _set_length('elbow')
    _set_length('wrist')
    base_skeleton['neck'] = offset_directions['neck'] * (body_lengths['neck']/normalization)

    kpts['offset_directions'] = offset_directions
    kpts['base_skeleton'] = base_skeleton
    kpts['normalization'] = normalization

    return

#calculate the rotation of the root joint with respect to the world coordinates
def get_hips_position_and_rotation(frame_pos, root_joint = 'hips', root_define_joints = ['lefthip', 'neck']):

    #root position is saved directly
    root_position = frame_pos[root_joint]

    #calculate unit vectors of root joint

    # x axis
    root_u = frame_pos[root_define_joints[0]] - frame_pos[root_joint]
    root_u = root_u/np.sqrt(np.sum(np.square(root_u)))

    # y axis
    root_v = frame_pos[root_define_joints[1]] - frame_pos[root_joint]
    root_v = root_v/np.sqrt(np.sum(np.square(root_v)))

    # z axis
    root_w = np.cross(root_u, root_v)



    #Make the rotation matrix
    C = np.array([root_u, root_v, root_w]).T
    thetaz,thetay, thetax = Decompose_R_ZXY(C)
    root_rotation = np.array([thetaz, thetax, thetay])

    return root_position, root_rotation

#calculate the rotation matrix and joint angles input joint
def get_joint_rotations(joint_name, joints_hierarchy, joints_offsets, frame_rotations, frame_pos):

    _invR = np.eye(3)
    for i, parent_name in enumerate(joints_hierarchy[joint_name]):
        if i == 0: continue
        _r_angles = frame_rotations[parent_name]
        R = get_R_z(_r_angles[0]) @ get_R_x(_r_angles[1]) @ get_R_y(_r_angles[2])
        _invR = _invR@R.T

    b = _invR @ (frame_pos[joint_name] - frame_pos[joints_hierarchy[joint_name][0]])

    _R = Get_R2(joints_offsets[joint_name], b)
    tz, ty, tx = Decompose_R_ZXY(_R)
    joint_rs = np.array([tz, tx, ty])
    #print(np.degrees(joint_rs))

    return joint_rs

#helper function that composes a chain of rotation matrices
def get_rotation_chain(joint, hierarchy, frame_rotations):

    hierarchy = hierarchy[::-1]

    #this code assumes ZXY rotation order
    R = np.eye(3)
    for parent in hierarchy:
        angles = frame_rotations[parent]
        _R = get_R_z(angles[0])@get_R_x(angles[1])@get_R_y(angles[2])
        R = R @ _R

    return R

#calculate the joint angles frame by frame.
def calculate_joint_angles(kpts):

    #set up emtpy container for joint angles
    for joint in kpts['joints']:
        kpts[joint+'_angles'] = []

    for framenum in range(kpts['hips'].shape[0]):

        #get the keypoints positions in the current frame
        frame_pos = {}
        for joint in kpts['joints']:
            frame_pos[joint] = kpts[joint][framenum]

        root_position, root_rotation = get_hips_position_and_rotation(frame_pos)

        frame_rotations = {'hips': root_rotation}

        #center the body pose
        for joint in kpts['joints']:
            frame_pos[joint] = frame_pos[joint] - root_position

        #get the max joints connectsion
        max_connected_joints = 0
        for joint in kpts['joints']:
            if len(kpts['hierarchy'][joint]) > max_connected_joints:
                max_connected_joints = len(kpts['hierarchy'][joint])

        depth = 2
        while(depth <= max_connected_joints):
            for joint in kpts['joints']:
                if len(kpts['hierarchy'][joint]) == depth:
                    joint_rs = get_joint_rotations(joint, kpts['hierarchy'], kpts['offset_directions'], frame_rotations, frame_pos)
                    parent = kpts['hierarchy'][joint][0]
                    frame_rotations[parent] = joint_rs
            depth += 1

        #for completeness, add zero rotation angles for endpoints. This is not necessary as they are never used.
        for _j in kpts['joints']:
            if _j not in list(frame_rotations.keys()):
                frame_rotations[_j] = np.array([0.,0.,0.])

        #update dictionary with current angles.
        for joint in kpts['joints']:
            kpts[joint + '_angles'].append(frame_rotations[joint])


    #convert joint angles list to numpy arrays.
    for joint in kpts['joints']:
        kpts[joint+'_angles'] = np.array(kpts[joint + '_angles'])
        #print(joint, kpts[joint+'_angles'].shape)

    return
