#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('TkAgg')
import sys
import os
import math
import time
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename


import numpy as np
import pandas as pd
from scipy import spatial

# Calculate features

# calculate distance
def calc_dist(data):
    data_head = data[:,0:2]
    dist_head = np.diagonal(spatial.distance.squareform(spatial.distance.pdist(data_head,'euclidean')),1)

    data_neck = data[:,2:4]
    dist_neck = np.diagonal(spatial.distance.squareform(spatial.distance.pdist(data_neck,'euclidean')),1)
    distmat = np.vstack((dist_head,dist_neck))

    data_body = data[:,4:6]
    dist_body = np.diagonal(spatial.distance.squareform(spatial.distance.pdist(data_body,'euclidean')),1)
    distmat = np.vstack((distmat,dist_body))

    data_rear = data[:,6:8]
    dist_rear = np.diagonal(spatial.distance.squareform(spatial.distance.pdist(data_rear,'euclidean')),1)
    distmat = np.vstack((distmat,dist_rear))

    data_tail = data[:,8:10]
    dist_tail = np.diagonal(spatial.distance.squareform(spatial.distance.pdist(data_tail,'euclidean')),1)
    distmat = np.vstack((distmat,dist_tail))

    distmat = np.transpose(distmat)
    return distmat

# calculate velocity
def calc_velocity(distmat):
    nframes = 9000
    fps = 15
    vidduration = nframes/fps
    timestep = vidduration/nframes
    velmat = distmat/timestep
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

# calculate logical array 0000 or 11111, if animal moved in last 10 sec move to 1 else 0
def calc_relative_motion():
    ...

# Temporal what was happening 5 min earlier
def calc_previous_state():
    tol_val = 0.01
    ...


# calculate distance relative to body of animal
def calc_dist_from_body(data):
    pos_head = data[:,0:2] - data[:,4:6]
    pos_neck = data[:,2:4] - data[:,4:6]
    pos_body = data[:,4:6] - data[:,4:6]
    pos_rear = data[:,6:8] - data[:,4:6]
    pos_tail = data[:,8:10] - data[:,4:6]
    distfrombody = np.hstack((pos_head,pos_neck,pos_body,pos_rear,pos_tail))
    return distfrombody

def do_lda(data):
    ...

def do_tsnei(data, ncomponents, verbosity, iperplexity, maxiter):
    from sklearn.manifold import TSNE
    from sklearn.metrics import pairwise_distances

    distance_matrix = pairwise_distances(data, data, metric='cosine', n_jobs=-1)

    tsne = TSNE(n_components=ncomponents, verbose=verbosity, perplexity=iperplexity, n_iter=maxiter, metric='precomputed')
    #tsne_results = tsne.fit_transform(data)
    tsne_results = tsne.fit_transform(distance_matrix)
    return tsne_results

def do_pca(data):
    mu = data.mean(axis=0)
    data = data - mu
    data = (data - mu)/data.std(axis=0)  # Uncommenting this reproduces mlab.PCA results
    eigenvectors, eigenvalues, V = np.linalg.svd(data.T, full_matrices=False)
    projected_data = np.dot(data, eigenvectors)
    sigma = projected_data.std(axis=0).mean()
    print(eigenvectors)
    return projected_data, eigenvectors, eigenvalues, V, sigma

def plot_tsne_out(out_tsne):
    from matplotlib import pyplot as plt
    plt.scatter(out_tsne[:,0], out_tsne[:,1])
    plt.show()

def plot_tsne_out_3d(out_tsne):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = out_tsne[:,0]
    y = out_tsne[:,1]
    z = out_tsne[:,2]

    ax.scatter(x, y, z, c='r', marker='o')
    #ax.set_xlabel('X Label')
    #ax.set_ylabel('Y Label')
    #ax.set_zlabel('Z Label')
    plt.show()


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
             ax.scatter(x[i], y[i], [], label = l)
             
        ax.legend()
        plt.show()

def clust_features(feature_matrix,label):
	#pull out indices of clusters
	#blue cluster
	cluster_1_ind = np.where(labels==0)
	cluster_1_features = feature_mat[cluster_1_ind[0],:]
	#red cluster
	cluster_2_ind = np.where(labels==3)
	cluster_2_features = feature_mat[cluster_2_ind[0],:]
	#pink cluster
	cluster_3_ind = np.where(labels==6)
	cluster_3_features = feature_mat[cluster_3_ind[0],:]    
	#green cluster
	cluster_4_ind = np.where(labels==2)
	cluster_4_features = feature_mat[cluster_4_ind[0],:]
	#purple cluster
	cluster_5_ind = np.where(labels==4)
	cluster_5_features = feature_mat[cluster_5_ind[0],:]
	#orange cluster
	cluster_6_ind = np.where(labels==1)
	cluster_6_features = feature_mat[cluster_6_ind[0],:]
	#brown cluster
	cluster_7_ind = np.where(labels==5)
	cluster_7_features = feature_mat[cluster_7_ind[0],:]

	#view means of each cluster for each feature
	meandist = np.array([np.mean(cluster_1_features[:,0:5],axis=0),np.mean(cluster_2_features[:,0:5],axis=0),np.mean(cluster_3_features[:,0:5],axis=0),np.mean(cluster_4_features[:,0:5],axis=0),np.mean(cluster_5_features[:,0:5],axis=0),np.mean(cluster_6_features[:,0:5],axis=0),np.mean(cluster_7_features[:,0:5],axis=0)])
	clust_dist = pd.DataFrame(meandist,columns=['Head_dist', 'Neck_dist', 'Body_dist', 'Rear_dist', 'Tail_dist'])
	clust_dist.insert(0,'Cluster',pd.Series(['Blue','Red','Pink','Green','Purple','Orange','Brown']))

	meanvel = np.array([np.mean(cluster_1_features[:,5:10],axis=0),np.mean(cluster_2_features[:,5:10],axis=0),np.mean(cluster_3_features[:,5:10],axis=0),np.mean(cluster_4_features[:,5:10],axis=0),np.mean(cluster_5_features[:,5:10],axis=0),np.mean(cluster_6_features[:,5:10],axis=0),np.mean(cluster_7_features[:,5:10],axis=0)])
	clust_vel = pd.DataFrame(meanvel,columns=['Head_vel', 'Neck_vel', 'Body_vel', 'Rear_vel', 'Tail_vel'])
	clust_vel.insert(0,'Cluster',pd.Series(['Blue','Red','Pink','Green','Purple','Orange','Brown']))

	meanaccel = np.array([np.mean(cluster_1_features[:,10:15],axis=0),np.mean(cluster_2_features[:,10:15],axis=0),np.mean(cluster_3_features[:,10:15],axis=0),np.mean(cluster_4_features[:,10:15],axis=0),np.mean(cluster_5_features[:,10:15],axis=0),np.mean(cluster_6_features[:,10:15],axis=0),np.mean(cluster_7_features[:,10:15],axis=0)])
	clust_accel = pd.DataFrame(meanaccel,columns=['Head_accel', 'Neck_accel', 'Body_accel', 'Rear_accel', 'Tail_accel'])
	clust_accel.insert(0,'Cluster',pd.Series(['Blue','Red','Pink','Green','Purple','Orange','Brown']))

	return clust_dist, clust_vel, clust_accel



if __name__ == '__main__':
    from sklearn.decomposition import PCA
    # Constants
    nfeatures = 5
    feature_coords = 3
    lcal = 0
    lsave = 0
    
    # open file browser
    # if len(sys.argv) > 1:
    #     dataname = (sys.argv[1])
    # else:
    #     Tk().withdraw() 
    #     dataname = askopenfilename() # open file browser
    dataname = '/Users/sbrunwas/Documents/HengenLab/DeepLabCut/DeepLabCut_v1/videos/e3v8103_day_subclipDeepCut_resnet50_ratmovementNov5shuffle1_22500.h5'
    
    df = pd.read_hdf(dataname)
    
    # structure of dataframe 
    # scorer    Bodyparts   x   y   likelihood
    #            head
    #            neck
    #            body
    #            rear
    #            tail
    
    
    # List columns
    #list(df)
    
    #df.columns
    
    print(df.columns.contains)
    # <bound method MultiIndex.__contains__ of MultiIndex(levels=[['DeepCut_resnet50_ratmovementNov5shuffle1_22500'], ['body', 'head', 'neck', 'rear', 'tail'], ['likelihood', 'x', 'y']],
    #                   labels=[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 2, 2, 2, 0, 0, 0, 3, 3, 3, 4, 4, 4], [1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0]],
    #                   names=['scorer', 'bodyparts', 'coords'])>
    
    
    #df['DeepCut_resnet50_ratmovementNov5shuffle1_22500']['body'][['x', 'y', 'likelihood']]
    
    # Convert multiindex data frame to numpy array
    o = df.values
    
    # add a preprocessing function here to sort data based on likelihood
    # ...
    
    # remove likelihood column
    m = np.delete(o, np.arange(feature_coords-1, nfeatures*feature_coords , feature_coords), axis=1)
    
    
    #get features for PCA/LDA
    if lcal == 1:
        distmat = calc_dist(m)
        if lsave == 1:
            np.save('distmat.npy', distmat)
        velmat = calc_velocity(distmat)
        if lsave == 1:
            np.save('velmat.npy', velmat)
        accelmat = calc_acceleration(velmat)
        if lsave == 1:
            np.save('accelmat.npy', accelmat)
        #distfrombody = calc_dist_from_body(m)
    else:
        distmat = np.load('/Volumes/HlabShare/Animal_tracking_deeplabcut/distmat.npy')
        velmat = np.load('/Volumes/HlabShare/Animal_tracking_deeplabcut/velmat.npy')
        accelmat = np.load('/Volumes/HlabShare/Animal_tracking_deeplabcut/accelmat.npy')

    
    # Feature matrix
    feature_mat = np.hstack((distmat, velmat, accelmat))
    print(feature_mat.shape)
    
    # kmeans for labels
    if lcal == 1:
	    labels = do_kmeans(feature_mat, 4)
	    if lsave == 1:
	        np.save('labels.npy', labels)
	else:
    	labels = np.load('/Volumes/HlabShare/Animal_tracking_deeplabcut/labels_8.npy')
    
    # do pca
    #reduced_data = PCA(n_components=15).fit_transform(feature_mat)
    proj_data, eig_vec, eig_val, V, sigma  = do_pca(feature_mat)
    #plot_with_labels(proj_data, labels)
    #proj_data, eigenvectors, eigenvalues, V, sigma = do_pca(feature_mat)
    
    # t-sne
    # def do_tsnei(data, ncomponents, verbosity, iperplexity, maxiter):
    if lcal == 1:
    	r_tsne = do_tsnei(proj_data, 2, 1, 40, 1000)
    	if lsave == 1:
        	np.save('r_tsne.npy', r_tsne)
    else:
    	r_tsne = np.load('/Volumes/HlabShare/Animal_tracking_deeplabcut/r_tsne_8.npy')
    
    # plot
    # plot_tsne_out(r_tsne)
    # plot 3d plot_tsne_out_3d(r_tsne)

    plot_with_labels(r_tsne, labels)

    #Assign features to clusters
    clust_dist, clust_vel, clust_accel =clust_features(feature_mat,labels)

