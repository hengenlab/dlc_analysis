import os
import numpy as np
import pandas as pd
import glob
import operator
from scipy import spatial
from matplotlib import pyplot as plt
from samb_work import videotimestamp

def get_pos(videoh5file,cutoff):
    df = pd.read_hdf(videoh5file)
    o = df.values
    # delfeafure = 1
    # o = np.delete(o,range(delfeafure*3-3,delfeafure*3),axis=1)
    nfeatures = int(np.shape(o)[1]/3)
    p_cutoff = cutoff
    for i in range(0,nfeatures):
        pro = o[:,3*i+2]
        lowpro =  np.where(pro <  p_cutoff)[0]
        o[lowpro,3*i] = np.nan
        o[lowpro,3*i+1] = np.nan
    m = np.delete(o, np.arange(2, nfeatures*3 , 3), axis=1)
    return m

def get_timeandpose(nfeatures,vidfile,cutoff):
    posfiles = []
    for filename in glob.glob('*.h5'):
        posfiles.append(filename)

    parsed_fls = [int(fl.split('Deep')[0].split('-')[1].replace('T','')) for fl in posfiles]
    file_dict = {}
    for idx, fl in enumerate(posfiles):
    	file_dict[fl] = parsed_fls[idx]
    sorted_dict = sorted(file_dict.items(), key=operator.itemgetter(1))
    sorted_posfiles = [t[0] for t in sorted_dict]
    sorted_timestamps = [t[1] for t in sorted_dict]

    posfiles_24hr = sorted_posfiles

    pos_total = np.zeros([1,2*nfeatures])     #  need to refine

    for i in np.arange(np.shape(posfiles_24hr)[0]):
    	filename = posfiles_24hr[i]
    	pos_i = get_pos(filename,cutoff)
    	pos_total = np.vstack((pos_total,pos_i))
    pos_total = np.delete(pos_total,0,0)

    timevect_samples = np.arange(pos_total.shape[0])
    timevect_nansec = timevect_samples/15
    #Get video start time

    video_start = (videotimestamp.vidtimestamp(vidfile))/(1000*1000*1000)

    timevect_corrected = video_start + timevect_nansec
    #combine time array and pos mat
    time_and_pos = np.vstack((timevect_corrected,pos_total.T)).T

    return time_and_pos


###  To load data from several hours
os.chdir('/Users/xuyifan/neural_data/H5-0402')
vidfile = '/Users/xuyifan/neural_data/Digitaldata/Digital_1_Channels_int64_2019-04-02_11-47-15.bin'   # digital file
cutoff = 0.8   # cutoff of likelihood of estimation
nfeatures = 11
time_and_pose = get_timeandpose(nfeatures,vidfile,cutoff)

### To load data from single hdf_file
os.chdir('/Users/xuyifan/neural_data/newcam')
dataname = 'e3v8102-20190330T1403-1423DeepCut_resnet50_wb45testMar30shuffle1_400000.h5'
cutoff = 0.8   # cutoff of likelihood of estimation
m = get_pos(dataname,cutoff)
