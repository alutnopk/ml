# Name: Kartik Pontula
# Roll No: 20CS10031
# Serial No: 12
# Project Code: HCHC-DS
# Project Name: Hospital Charge Category using Single Linkage Divisive (Top-Down) Clustering Technique
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.utils import resample
from scipy.spatial import ConvexHull # used to find cluster diameter efficiently
import time

def cosine_distance(X1, X2):
	return 1.0 - np.dot(X1, X2) / (np.linalg.norm(X1) * np.linalg.norm(X2))

def single_linkage_distance(X, ix1, ix2):
	min_dist = float('inf')
	for i in ix1:
		for j in ix2:
			min_dist = min(min_dist, cosine_distance(X[i], X[j]))
	return min_dist
	

def cluster_diameter(X, ix):
	# O(n^2) implementation here:
	diam = 0
	for i_ in range(len(ix)):
		for j_ in range(i_+1, len(ix)):
			diam = max(diam, cosine_distance(X[ix[i_]], X[ix[j_]]))
	return diam

	# # efficient implementation by finding farthest points on the convex hull, O(n log n)
	# X_c = X[ix]
	# hull = ConvexHull(X_c)
	# diam = 0
	# for i in range(len(hull.vertices)):
	# 	for j in range(i+1, len(hull.vertices)):
	# 		pi = X_c[hull.vertices[i]]
	# 		pj = X_c[hull.vertices[j]]
	# 		cos_dist = cosine_distance(pi, pj)
	# 		diam = max(diam, cos_dist)
	# return diam

def k_means_clusterization(X, k = 3, epochs = 20):
	cluster_map = {}
	centres = X[np.random.choice(X.shape[0], k, replace=False)]
	print(f"Initial centres: \n{centres}")
	for _ in range(epochs):
		for c in centres:
			cluster_map[tuple(c)] = []
		# classify all points in X to their closest centre using cosine similarity
		for idx in range(len(X)):
			# find the closest centre and add to it
			closest_centre = min(centres, key=lambda c: cosine_distance(c, X[idx]))
			cluster_map[tuple(closest_centre)].append(idx)
		# replace each centre with the cluster mean
		for c in centres:
			new_c = np.mean(X[cluster_map[tuple(c)]], axis=0)
			cluster_map[tuple(new_c)] = cluster_map.pop(tuple(c))
		centres = list(cluster_map.keys())
		# print cluster count
		print(f"\nEpoch {_ + 1}:")
		print(f"Cluster centres\t\t\t\t\t\t\tCluster sizes")
		for c, ix in cluster_map.items():
			print(f"{c}\t{len(ix)}")
	return cluster_map

def silhouette_coefficient(X, cluster_map):
	silhouette_values = []
	# iterate over each point
	for i in range(len(X)):
		# find cluster it belongs to
		current_centre = ()
		for c, ix in cluster_map.items():
			if X[i] in X[ix]:
				current_centre = c
				break
		# find (a)
		a = np.mean([cosine_distance(X[i], X[j]) for j in cluster_map[current_centre]])
		# find nearest neighbor cluster
		nbr_centres = list(cluster_map.keys())
		nbr_centres.remove(current_centre)
		nearest_nbr_centre = min(nbr_centres, key = lambda c: cosine_distance(np.array(c), current_centre))
		# find (b)
		b = np.mean([cosine_distance(X[i], X[j]) for j in cluster_map[nearest_nbr_centre]])
		# calculate s_i
		s_i = (b - a) / max(a, b)
		silhouette_values.append(s_i)
	return np.mean(silhouette_values)

def divisive_hierarchial_clusterization(X, k = 3):
	cluster_map = {}
	cluster_map[tuple(np.mean(X, axis=0))] = [i for i in range(len(X))]

	while len(cluster_map) < k:
		# find the cluster with the largest diameter
		target_cluster_centre = max(list(cluster_map.keys()), key = lambda c: cluster_diameter(X, cluster_map[c]))
		target_cluster_list = cluster_map[target_cluster_centre]
		# find the split point within this cluster (i.e. farthest point from cluster)
		max_avg_dist = 0.0
		split_point = -1
		for i in target_cluster_list:
			avg_dist_i = np.mean([cosine_distance(X[i], X[j]) for j in target_cluster_list])
			if avg_dist_i > max_avg_dist:
				max_avg_dist = avg_dist_i
				split_point = i
		
		# # can be done more efficiently by doing this instead
		# split_point = max(target_cluster_list, key = lambda i: cosine_distance(X[i], np.array(target_cluster_centre)))

		# split the cluster about this index
		print(f"{len(cluster_map)} cluster(s) present, splitting cluster with centre {target_cluster_centre} using point at index {split_point}")
		subcluster_1_list = [split_point]
		subcluster_2_list = target_cluster_list # no need to copy perhaps
		subcluster_2_list.remove(split_point)
		# pick each point from the original cluster, and assign it to the closer of both subclusters
		for i in target_cluster_list:
			print(".", end="")
			subcluster_1_centre = np.mean(X[subcluster_1_list], axis=0)
			subcluster_2_centre = np.mean(X[subcluster_2_list], axis=0)
			if i in subcluster_1_list:
				continue
			subcluster_2_list.remove(i)
			# assign i to the closer subcluster
			# # uncomment below for a more efficient if-statement
			# if cosine_distance(subcluster_1_centre, X[i]) < cosine_distance(subcluster_2_centre, X[i]):
			if single_linkage_distance(X, [split_point], [i]) < single_linkage_distance(X, subcluster_2_list, [i]):
				subcluster_1_list.append(i)
			else:
				subcluster_2_list.append(i)
		# Update the cluster map
		del cluster_map[target_cluster_centre]
		subcluster_1_centre = tuple(np.mean(X[subcluster_1_list], axis=0))
		subcluster_2_centre = tuple(np.mean(X[subcluster_2_list], axis=0))
		cluster_map[subcluster_1_centre] = subcluster_1_list
		cluster_map[subcluster_2_centre] = subcluster_2_list
		print("weee")
	
	return cluster_map

def main():
	start_time = time.time()
	# parse and preprocess input
	df = pd.read_csv('./dataset/hospital.csv')
	target_cols = ['Average Covered Charges', 'Average Total Payments', 'Average Medicare Payments']
	df = df[target_cols].map(lambda x: float(x.replace('$', '')))
	X = df.values
	# perform k-means clustering
	sil_scores = {}
	for k in range(3, 7):
		cluster_map = k_means_clusterization(X, k, epochs = 20)
		# calculate silhouette score for a random sample of the data
		labels = np.empty(X.shape[0], dtype=np.int64)
		for i, cluster_indices in enumerate(cluster_map.values()):
			labels[cluster_indices] = i
		print(f"Calculating Silhouette coefficient for k = {k}")
		X_sample, labels_sample = resample(X, labels, n_samples = 40000, random_state = 42)
		score = silhouette_score(X_sample, labels_sample, metric='cosine')
		print(f'Silhouette score: {score}')
		sil_scores[k] = score
		if k == 3: # kmeans.txt
			with open('kmeans.txt', 'w') as f:
				for vals in cluster_map.values():
					vals = sorted(list(vals))
					f.write(",".join([str(x) for x in vals]))
						
	# # save results to file
	# for k, v in sil_scores.items():
	# 	print(f"k = {k}, silhouette score = {v}")

	# with open('silhouette_scores_kmeans.txt', 'w') as f:
	# 	for k, v in sil_scores.items():
	# 		f.write(f"k = {k}, silhouette score = {v}\n")
	# # perform divisive hierarchical clustering
	# mp = divisive_hierarchial_clusterization(X, 3)
	# # save results to file
	# with open('divisive.txt', 'w') as f:
	# 	for k, v in mp.items():
	# 		f.write(f"k = {k}, silhouette score = {v}\n")

	end_time = time.time()
	print(f"Total running time: {end_time - start_time} seconds")

if __name__ == '__main__':
	main()