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

# calculate the movement of each body part between frames  # for movement analysis
def calc_dis(data):
    headstage = data[:,0:2]
    Lear = data[:,2:4]
    Rear = data[:,4:6]

    # add more features here

    HS = np.diff(headstage,axis =0)
    disHS = np.hypot(HS[:,0],HS[:,1])

    LE = np.diff(Lear,axis =0)
    disLE = np.hypot(LE[:,0],LE[:,1])

    RE = np.diff(Rear,axis =0)
    disRE = np.hypot(RE[:,0],RE[:,1])

    distmat = np.vstack((disHS,disLE,disRE))
    distmat = distmat.T

    return distmat

# calculate movement of x,y positions of nose and tail-base between frames  # for pose analysis
def calc_pose1(data):

    nose = data[:,4:6]
    tail = data[:,10:12]

    xymat = np.zeros([data.shape[0]-1,4])

    xymat[:,0] = np.diff(nose[:,0])
    xymat[:,1] = np.diff(nose[:,1])
    xymat[:,2] = np.diff(tail[:,0])
    xymat[:,3] = np.diff(tail[:,1])

    return xymat

# calculate distance between body parts in each frame 
def calc_pose2(data):

    ear1 = data[:,0:2]
    ear2 = data[:,2:4]
    nose = data[:,4:6]
    tail = data[:,6:8]

    posedismat = np.zeros([data.shape[0],6])

    for i in range(0,data.shape[0]):
        posett = np.vstack((ear1[i,:],ear2[i,:],nose[i,:],tail[i,:]))
        posedismat[i,:] = spatial.distance.pdist(posett,'euclidean')

    return posedismat

# calculate relative position between body parts in each frame 
def calc_pose3(data):

    ear1 = data[:,0:2]
    ear2 = data[:,2:4]
    nose = data[:,4:6]
    tail = data[:,6:8]

    posemat = np.zeros([data.shape[0],10])
    
    posemat[:,0:2] = ear1 - ear2
    posemat[:,2:4] = ear1 - nose
    posemat[:,4:6] = ear2 - nose
    posemat[:,6:8] = ear1 - tail
    posemat[:,8:10] = ear2- tail

    return posemat

# calculate velocity
def calc_velocity(distmat):

    fps = 15
    velmat = distmat * fps
    return velmat

# calculate acceleration
def calc_acceleration(velmat):
    #make a matrix that is shifted vertically by 1
    rollmat = np.roll(velmat,1,axis=0)
    #make first row of shifted matrix 0 since it is the initial point in time
    rollmat[0] = 0
    #subtract the original velocity matrix from the shifted matrix
    accelmat = velmat-rollmat
    return accelmat

def do_tsnei(data, ncomponents, verbosity, iperplexity, maxiter):
    from sklearn.manifold import TSNE
    from sklearn.metrics import pairwise_distances

    distance_matrix = pairwise_distances(data, data, metric='cosine', n_jobs=-1)

    tsne = TSNE(n_components=ncomponents, verbose=verbosity, perplexity=iperplexity, n_iter=maxiter, metric='precomputed')
    #tsne_results = tsne.fit_transform(data)
    tsne_results = tsne.fit_transform(distance_matrix)
    return tsne_results

def do_pca(data,k):

    mu = data.mean(axis=0)
    data = data - mu
    data = (data - mu)/data.std(axis=0)

    covMatrix = np.cov(data,rowvar = 0)

    eigenvectors, eigenvalues, V = np.linalg.svd(covMatrix, full_matrices=False)
    U = eigenvectors[:,0:k]
    projected_data = np.dot(data, U)

    return projected_data

def do_kmeans(data, k):
    from sklearn.cluster import KMeans
    #seed = 0
    #km = KMeans(n_clusters=k, 'random', n_init=10, max_iter=1000, random_state=seed)
    km = KMeans(n_clusters=k, n_init=100, max_iter=10000, tol=1e-6)
    km.fit(data)
    y_cluster_kmeans = km.predict(data)
    return y_cluster_kmeans

def plot_with_labels(out_tsne, label):
        import numpy as np
        from matplotlib import pyplot as plt
        x = out_tsne[:,0]
        y = out_tsne[:,1]

        fig, ax = plt.subplots()
        for l in np.unique(label):
             i = np.where(label == l)
             ax.scatter(x[i], y[i], s =5,alpha=0.6,label = l)

        ax.legend()
        plt.show()


################################################ 
# To load data from single hdf_file, example for pose analysis based on ears, nose and tail-base

os.chdir('/Users/xuyifan/neural_data/newcam')
dataname = 'e3v8102-20190330T1403-1423DeepCut_resnet50_wb45testMar30shuffle1_400000.h5'
cutoff = 0.8     # cutoff of likelihood of estimation
m = get_pos(dataname,cutoff)

delfeafure = 1       # delete the headstage (not shown in this video)
m = np.delete(m,range(delfeafure*2-2,delfeafure*2),axis=1)

frames = np.arange(m.shape[0])
mm = np.vstack((frames,m.T)).T      ## with frame index

posedata =np.zeros([m.shape[0],8])
posedata[:,0:6] = m[:,0:6].copy()
posedata[:,6:8] = m[:,10:12].copy()  # get position of two ears, nose and tail-base

inde = mm[:,0].copy()
xymat = calc_pose1(m)      # get speed of x,y positions of nose and tail-junc
pose = np.delete(posedata,np.where(np.isnan(posedata))[0],axis=0)   # delete "nan" data point (based on cutoff)
index = np.delete(inde,np.where(np.isnan(posedata))[0],axis=0)
xymat = xymat[index.astype(int) - 1,:]
nan = np.where(np.isnan(xymat))[0]
xymat = np.delete(xymat,nan,axis=0)
pose= np.delete(pose,nan,axis=0)
index = np.delete(index,nan,axis=0)

posedismat = calc_pose2(pose)     # calculate distance between body parts in each frame
posemat = calc_pose3(pose)        # calculate relative position between body parts in each frame 

movemat = np.hstack((xymat,posemat))

movemat2 = np.vstack((xymat.T,posemat[:,5]))    # x,y speed of nose and tail-base, relative distance between nose and tail-base
movemat2 = movemat2.T

proj_data  = do_pca(movemat2,3)
labels =  do_kmeans(movemat2,4)
r_tsne = do_tsnei(proj_data, 2, 1, 40, 10000)

plot_with_labels(r_tsne, labels)

### analyze pose based on tsne clustering data
cluster = np.where((r_tsne[:,0] < -70) & (r_tsne[:,0] > -100) &(r_tsne[:,1] < 40) & (r_tsne[:,1] > 30))

videofilename ='/Users/xuyifan/neural_data/newcam/e3v8102-20190330T1403-1423DeepCut_resnet50_wb45testMar30shuffle1_400000_labeled.mp4'
lstream = 0
v = ntk.NTKVideos(videofilename, lstream)
frame_num = 100 
frame_num = index[50]     # get the frame_num from cluster
v.grab_frame_num(frame_num)

###  3D plot of PCA proj_data
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = plt.subplot(111, projection='3d')
x,y,z = proj_data[:,0],proj_data[:,1],proj_data[:,2]
ax.scatter(x, y, z, c='r',alpha=0.2)
ax.set_zlabel('PC3')
ax.set_ylabel('PC2')
ax.set_xlabel('PC1')
plt.show()

################################################ 
#  To load data from several hours, example for movement analysis combined with neural elec data
os.chdir('/Users/xuyifan/neural_data/H5-0402')
vidfile = '/Users/xuyifan/neural_data/Digitaldata/Digital_1_Channels_int64_2019-04-02_11-47-15.bin'   # digital file
cutoff = 0.8     # cutoff of likelihood of estimation
nfeatures = 11
time_and_pose = get_timeandpose(nfeatures,vidfile,cutoff)

m1 = time_and_pose[:,1:7].copy()
distmat = calc_dis(m1)
distance = np.nanmean(distmat,axis=1)

os.chdir('/Users/xuyifan/neural_data/t_04-02')
rsuspikes = np.load('rsuspikes.npy')
fsspikes = np.load('fsspikes.npy')    # spiking time of each unit after sorting and qual_check
rsunum = rsuspikes.shape[0]
fsnum = fsspikes.shape[0]

def calc_corr(binsize):
    
    slice = binsize*15
    slicenum = np.int(round(distance.shape[0]/slice)-1)

    meandis = np.zeros([slicenum,])
    for i in range(0,slicenum):
        meandis[i,]=np.nanmean(distance[i*slice:(i+1)*slice,])       # 0 [0:150]  1 [150:300]

    starttime = time_and_pose[0,0]
    endtime = starttime + slicenum*binsize
    edges = np.arange(starttime,endtime+binsize,binsize)

    rsutrace = np.zeros([rsunum,edges.shape[0]-1])
    for i in range(0,rsunum):
        a = np.histogram(rsuspikes[i],edges)
        rsutrace[i,:]=a[0]/binsize

    fstrace = np.zeros([fsnum,edges.shape[0]-1])
    for j in range(0,fsnum):
        a = np.histogram(fsspikes[j],edges)
        fstrace[j,:]=a[0]/binsize

    meanrsu = np.mean(rsutrace,axis=0)
    meanfs = np.mean(fstrace,axis=0)

    return meandis,meanrsu,meanfs,rsutrace,fstrace

binsize = 60
meandis,meanrsu,meanfs,rsutrace,fstrace = calc_corr(binsize)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(meanrsu,'r')
ax1.set_ylabel('Firing Rate (Hz)', fontsize=15)
ax1.yaxis.label.set_color('red')

ax2 = ax1.twinx()
ax2.plot(meandis,'b')
ax2.set_ylabel('Movement (pixel)',fontsize=15)
ax2.yaxis.label.set_color('blue')
plt.show()

from scipy.stats.stats import pearsonr
rsu_corr = np.zeros([rsutrace.shape[0],])
for i in range(0,rsutrace.shape[0]):
    rsu_corr[i,] = pearsonr(meandis,rsutrace[i,:])[0]

fs_corr = np.zeros([fstrace.shape[0],])
for j in range(0,fstrace.shape[0]):
    fs_corr[j,] = pearsonr(meandis,fstrace[j,:])[0]

bins = np.linspace(-1,1,11)
plt.hist(fs_corr,bins,rwidth=0.5)
plt.xticks(bins)
plt.show()

bins = np.linspace(-1,1,11)
plt.hist(rsu_corr,bins,rwidth=0.5)
plt.xticks(bins)
plt.show()
