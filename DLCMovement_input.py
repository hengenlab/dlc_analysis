#Save time and motion vectors
# from samb_work import videotimestamp
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import scipy.signal as sig
from samb_work import videotimestamp
import operator
import os
import glob
import statistics as st
import scipy.stats as stats
from scipy import signal
import matplotlib.patches as patches 

#first argument = name of h5 file to be processed
#second argument = timeestamp of video save, as obtained from videotimestamp.vidtimestamp()

def get_pos(videoh5file):
	###GET POSITION VECTOR
	df = pd.read_hdf(videoh5file)
	#f = h5py.File(file_name, 'r')
	#group = f['df_with_missing']
	#data = group['table'].value

	nfeatures = 2
	feature_coords = 3

	o = df.values

	# remove likelihood column
	m = np.delete(o, np.arange(feature_coords-1, nfeatures*feature_coords , feature_coords), axis=1)
	return(m)
def group_consecutives(vals, step=1):
    """Return list of consecutive lists of numbers from vals (number list)."""
    run = []
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result

def get_movement(vidfile, first_vid = 1, digifile = False, num_labels = 1):
	fig1, ax1 = plt.subplots(nrows = 1, ncols = 1, figsize = [15, 4])
	plt.ion()
	pos_total = np.zeros(num_labels*2)

	pos_i = get_pos(vidfile)
	pos_total = np.vstack((pos_total,pos_i))
	pos_total = np.delete(pos_total, (0), axis = 0)

	timevect_frame = np.arange(pos_total.shape[0])
	timevect_nansec = (timevect_frame/15) * 1000 * 1000 * 1000 # in nanosec
	#Get video start time
	if first_vid:
		video_start = videotimestamp.vidtimestamp(digifile)
	else:
		video_start = 0


	timevect_corrected = video_start + timevect_nansec
	#combine time array and pos mat
	time_and_pos = np.vstack((timevect_corrected,pos_total.T)).T

	#time_and_pos_rectime = time_and_pos[2480:,:]

	time = time_and_pos[:,0]
	time_sec = time*1e-9
	dt = time_sec[1]
	binsz = int(round(1/dt))

	x_head = time_and_pos[:,1]
	nans = np.where(np.isnan(x_head))
	dx_head = np.diff(x_head)
	y_head = time_and_pos[:,2]
	dy_head = np.diff(y_head)
	dxy_head = np.sqrt(np.square(dx_head) + np.square(dy_head))

	time_min = (time/60)[0:-1]

	#plt.subplot(2,1,1)

	dxy = np.append(dxy_head, np.float('nan'))
	while np.size(dxy) < 54000:
		dxy = np.append(dxy, np.float('nan'))
	rs_dxy = np.reshape(dxy,[int(np.size(dxy)/binsz), binsz])
	time_min = np.linspace(0, 60, np.size(dxy))
	
	#thresh_line = np.full(np.shape(downsampled_dxy), thresh)
	# plt.plot(time_min, thresh_line)
	#plt.subplot(2,1,2)
	#downsampled_dxy = signal.resample(dxy_head, int(np.size(dxy_head)/2))
	#time_ds = np.linspace(0, 60, np.size(downsampled_dxy))
	#plt.plot(time_min, dxy)
	med = np.median(rs_dxy, axis = 1)
	binned_dxy = np.mean(rs_dxy, axis = 1)
	#thresh = np.percentile(med[~np.isnan(med)], 50)
	x_vals = np.linspace(0,60,np.size(med))
	#plt.plot(x_vals, binned_dxy)
	plt.plot(x_vals, med)
	sorted_med = np.sort(med)
	idx = np.where(sorted_med>int(max(sorted_med)*0.05))[0][0]

	if idx == 0:
		thresh = sorted_med[idx] 
	#print(int(max(sorted_med)*0.50))
	else:
		thresh = np.nanmean(sorted_med[0:idx])
	# ymax = plt.gca().get_ylim()[1]
	# plt.figure()
	# plt.plot(sorted_med)
	# plt.plot([idx,idx],[0, ymax])
	#print(thresh)

	# moving = np.where(med > thresh)[0]
	moving = np.where(dxy > thresh)[0]
	h = plt.gca().get_ylim()[1]
	# consec = group_consecutives(np.where(med > thresh)[0])
	consec = group_consecutives(np.where(med > thresh)[0])
	for vals in consec:
		if len(vals)>5:
			x = x_vals[vals[0]]
			#x = time_min[vals[0]]
			y = 0
			width = x_vals[vals[-1]]-x
			#width = time_min[vals[-1]]-x
			rect = patches.Rectangle((x,y), width, h, color = '#b7e1a1', alpha = 0.5)
			ax1.add_patch(rect)
	plt.show()
	plt.title(vidfile)
	np.save(vidfile[0:-3]+'_movement_trace.npy', med)













