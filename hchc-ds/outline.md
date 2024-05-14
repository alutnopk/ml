# Synopsis

- learn the following:
	- k-means clustering
	- single linkage divisive (top-down) hierarchical clustering, complete linkage strategy

- conduct 3-means clustering.
	- distance: cosine similarity (cos theta)
	- initial means: k random distinct points
	- 20 iters
	- save info in file

- silhouette coefficient
	- for each data point, find a and b
	- a is avg distance to points in same cluster
	- b is avg distance to points in closest cluster
	- s_i = find (b - a) / max(a, b)
	- return the avg value of s_i for all points

Cluster centers                                                 Cluster sizes
(52847.41453515896, 9259.463547241472, 8004.959898347217)       59418
(29816.412478651575, 9753.918246149158, 8517.38490805729)       76352
(17422.709037552417, 10552.820842645277, 9496.101458508969)     27295

random_state = 42
k = 3, silhouette score = 0.6685947769006753
k = 4, silhouette score = 0.6595587936497189
k = 5, silhouette score = 0.6395637229208407
k = 6, silhouette score = 0.63113455178858

random state = 69420
k = 3, silhouette score = 0.6720396255846293
k = 4, silhouette score = 0.6614674946757273
k = 5, silhouette score = 0.6473525454270854
k = 6, silhouette score = 0.6265072985572774
Total running time: 630.2164981365204 seconds



def silhouette_coefficient(X, cluster_map):
	silhouette_values = []
	# iterate over each point
	for i in range(len(X)):
		# find cluster it belongs to
		current_center = ()
		for c, ix in cluster_map.items():
			if X[i] in X[ix]:
				current_center = c
				break
		# find (a)
		a = np.mean([cosine_distance(X[i], X[j]) for j in cluster_map[current_center]])
		# b = float('inf')
		# for c, ix in cluster_map.items():
		# 	if c == current_center:
		# 		continue
		# 	b = min(np.mean([cosine_distance(X[i], X[j]) for j in ix]), b)
		# find nearest neighbor cluster
		nbr_centers = list(cluster_map.keys())
		nbr_centers.remove(current_center)
		# print(nbr_centers)
		nearest_nbr_center = min(nbr_centers, key = lambda c: cosine_distance(np.array(c), current_center))
		# find (b)
		b = np.mean([cosine_distance(X[i], X[j]) for j in cluster_map[nearest_nbr_center]])
		# calculate s_i
		s_i = (b - a) / max(a, b)
		silhouette_values.append(s_i)
	return np.mean(silhouette_values)
