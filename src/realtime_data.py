import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
import plotly
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import cv2
from tqdm import tqdm
from pathlib import Path
import utils
import math

CL_video_path = Path(r'/videos/output0.mp4')
CL_data_path = Path(r'primary_features.csv')
CL_raw_data_path = Path(r'raw_features.csv')
CL_data = pd.read_csv(CL_data_path)
CL_raw_data = pd.read_csv(CL_raw_data_path)

features_L = ['trunk_fwd_flex', 'trunk_fwd_flex']

features_R = ['trunk_lat_flex', 'trunk_lat_flex']


features_speed = ['leftelbow_flex_w','rightelbow_flex_w','leftshoulder_flex_w','rightshoulder_flex_w',
 			'leftshoulder_abduc_w','rightshoulder_abduc_w', 'leftshoulder_rot_w', 'rightshoulder_rot_w',
			'trunk_fwd_flex_w', 'trunk_lat_flex_w']

features_acc = ['leftshoulder_flex_a', 'rightshoulder_flex_a',
			'leftshoulder_abduc_a', 'rightshoulder_abduc_a',
			'leftshoulder_rot_a', 'rightshoulder_rot_a', 'leftelbow_flex _a',
			'rightelbow_flex_a', 'trunk_fwd_flex_a', 'trunk_lat_flex_a']

features_naive = ['leftelbow_angle', 'rightelbow_angle', 'leftshoulder_angle',
       'rightshoulder_angle', 'hips_angle']

features_centroid = ['leftarm_centroid',
       'rightarm_centroid', 'arms_centroid', 'trunk_centroid']

# features=features_speed

events = {'shoulder_flex': [65, 98],
		'shoulder_rot': [99, 132],
		'shoulder_abduc': [262, 304],
		'elbow_flex': [133, 192],
		'trunk_fwd_flex': [192, 214],
		'trunk_lat_flex': [214, 233],
		'trunk_rot': [233, 249]}


FPS=30
fs = 3
SIZE = (fs*100,fs*100)

def realtime_plot_data(file_name, data, start, end):
	fig, ax = plt.subplots(len(features_L),2,figsize=(fs,fs))
	out = None
	step = 3
	for t in tqdm(range(start,end,step),desc='rendering'):
		
		for i in range(len(features_L)):
			min_max_left = min(data[features_L[i]][start:end]),max(data[features_L[i]][start:end])
			min_max_right = min(data[features_R[i]][start:end]),max(data[features_R[i]][start:end])

			min_max = min(min_max_left[0],min_max_right[0]),max(min_max_left[1],min_max_right[1])

			ax[i][0].set_ylim(min_max[0], min_max[1])
			ax[i][0].set_xlim(start, end)
			ax[i][0].set_title(features_L[i])
			ax[i][0].plot(data[features_L[i]][start:t],'k-')

			ax[i][1].set_ylim(min_max[0], min_max[1])
			ax[i][1].set_xlim(start, end)
			ax[i][1].set_title(features_R[i])
			ax[i][1].plot(data[features_R[i]][start:t],'k-')

		fig.tight_layout()
		fig.canvas.draw()
		img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
		h = int(math.sqrt(img.shape[0]//3))
		img = img.reshape((h,h,3))
		if out == None:
			out = cv2.VideoWriter(f'videos/{file_name}.avi',cv2.VideoWriter_fourcc(*'MJPG'), FPS, (h,h))
		#w, h = fig.canvas.get_width_height()
		#img = img.reshape((int(h), int(w), -1))
		# ax[i].cla()

		for i in range(len(features_L)):
			ax[i//2][i%2].cla()

		out.write(img)
	out.release

def trim_video(file_name,vid,start,end):
	out = cv2.VideoWriter(f'videos/{file_name}.avi',cv2.VideoWriter_fourcc(*'MJPG'), FPS, (1280,720))
	i=0
	while vid.isOpened():
		# print(i)
		ret,frame = vid.read()
		if i >start and i <end:
			if ret == True:
				out.write(frame)
		if i > end:
			break
		i+=1
	out.release


#for event in events.values():
event = 'trunk_lat_flex'
	#print(event[0])
realtime_plot_data(f'CL_{event}',CL_data, events[event][0]*FPS, events[event][1]*FPS)
#realtime_plot_data(f'CL_{event}',CL_data, 0,10)

#CL_video = cv2.VideoCapture(str(CL_video_path))
#trim_video(f'CL_raw_vid_{event}',CL_video,events[event][0]*FPS,events[event][1]*FPS)
#video_skeleton(f'CL_raw_skeleton_{event}', CL_raw_data, events[event][0], events[event][1])
#video_skeleton(f'CL_proj_skeleton_{event}', CL_data, events[event][0], events[event][1])