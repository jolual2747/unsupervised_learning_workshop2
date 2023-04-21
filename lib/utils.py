import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize
from numpy.linalg import norm
from random import normalvariate
from math import sqrt
from typing import Iterable
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.cm as cm
import math

# PCA from scratch
class PCA:
    def __init__(self, n_components):
        """Principal component analysis (PCA) implementation.
        Transforms a dataset of possibly correlated values into n linearly
        uncorrelated components. The components are ordered such that the first
        has the largest possible variance and each following component as the
        largest possible variance given the previous components. This causes
        the early components to contain most of the variability in the dataset.
        Parameters
        ----------
        n_components : int
        solver : 'eigen'
        """
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # center the data
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # compute the covariance matrix
        cov = np.cov(X, rowvar=False)

        # compute the eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # sort the eigenvalues and eigenvectors in decreasing order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # store the first n_components eigenvectors as the principal components
        self.components = eigenvectors[:, : self.n_components]

    def transform(self, X):
        # center the data
        X = X - self.mean

        # project the data onto the principal components
        X_transformed = np.dot(X, self.components)
        return X_transformed    
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class svd_scratch:
    def __init__(self, n_components=None):
        """
        SVD (Singular Value Decomposition): is a factorization of a matrix into 
        three matrices. U, S, VT.
        It's used as a data reduction method in machine learning
        Parameters
        ----------
        n_components: int
            number of sigular values to decompose
        """
        self.n_components = n_components
        
    def fit(self, X):
        U, sigma, VT = np.linalg.svd(X, full_matrices=False)
        self.U = U[:, :self.n_components]
        self.sigma = np.diag(sigma)[0:self.n_components,:self.n_components]
        self.VT = VT[:self.n_components, :]
        
    def fit_transform(self, X):
        self.fit(X)
        X_transformed = self.U @ self.sigma @ self.VT
        return X_transformed
    
    def transform(self, X):
        X_transformed = self.U @ self.sigma @ self.VT
        return X_transformed
    
class tsne:
    def __init__(self, n_components=2, perplexity=15.0, max_iter=50, momentum = 1.0, learning_rate=10,random_state=1234):
        """
        T-SNE: A t-Distributed Stochastic Neighbor Embedding implementation. Built based on https://github.com/nlml/tsne_raw
        It's a tool to visualize high-dimensional data. It converts
        similarities between data points to joint probabilities and tries
        to minimize the Kullback-Leibler divergence between the joint
        probabilities of the low-dimensional embedding and the
        high-dimensional data. 
        Parameters:
        ----------
        max_iter : int, default 300
        perplexity : float, default 15.0
        n_components : int, default 2
        """
        self.n_components = n_components
        self.perplexity = perplexity
        self.max_iter = max_iter    
        self.momentum = momentum
        self.lr = learning_rate
        self.seed=random_state

    def fit(self, X):
        self.Y = np.random.RandomState(self.seed).normal(0., 0.0001, [X.shape[0], self.n_components])
        self.Q, self.distances = self.q_tsne()
        self.P=self.p_joint(X)

    def transform(self, X):
        if self.momentum:
            Y_m2 = self.Y.copy()
            Y_m1 = self.Y.copy()

        for i in range(self.max_iter):

            # Get Q and distances (distances only used for t-SNE)
            self.Q, self.distances = self.q_tsne()
            # Estimate gradients with respect to Y
            grads = self.tsne_grad()

            # Update Y
            self.Y = self.Y - self.lr * grads

            if self.momentum:  # Add momentum
                self.Y += self.momentum * (Y_m1 - Y_m2)
                # Update previous Y's for momentum
                Y_m2 = Y_m1.copy()
                Y_m1 = self.Y.copy()
        return self.Y

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def p_joint(self, X):
        """Given a data matrix X, gives joint probabilities matrix.
        # Arguments
            X: Input data matrix.
        # Returns:
            P: Matrix with entries p_ij = joint probabilities.
        """
        def p_conditional_to_joint(P):
            """Given conditional probabilities matrix P, return
            approximation of joint distribution probabilities."""
            return (P + P.T) / (2. * P.shape[0])
        def calc_prob_matrix(distances, sigmas=None, zero_index=None):
            """Convert a distances matrix to a matrix of probabilities."""
            if sigmas is not None:
                two_sig_sq = 2. * np.square(sigmas.reshape((-1, 1)))
                return self.softmax(distances / two_sig_sq, zero_index=zero_index)
            else:
                return self.softmax(distances, zero_index=zero_index)
        # Get the negative euclidian distances matrix for our data
        distances = self.neg_squared_euc_dists(X)
        # Find optimal sigma for each row of this distances matrix
        sigmas = self.find_optimal_sigmas()
        # Calculate the probabilities based on these optimal sigmas
        p_conditional = calc_prob_matrix(distances, sigmas)
        # Go from conditional to joint probabilities matrix
        self.P = p_conditional_to_joint(p_conditional)
        return self.P


    def find_optimal_sigmas(self):
        """For each row of distances matrix, find sigma that results
        in target perplexity for that role."""
        def binary_search(eval_fn, target, tol=1e-10, max_iter=10000,
                        lower=1e-20, upper=1000.):
            """Perform a binary search over input values to eval_fn.
            # Arguments
                eval_fn: Function that we are optimising over.
                target: Target value we want the function to output.
                tol: Float, once our guess is this close to target, stop.
                max_iter: Integer, maximum num. iterations to search for.
                lower: Float, lower bound of search range.
                upper: Float, upper bound of search range.
            # Returns:
                Float, best input value to function found during search.
            """
            for i in range(max_iter):
                guess = (lower + upper) / 2.
                val = eval_fn(guess)
                if val > target:
                    upper = guess
                else:
                    lower = guess
                if np.abs(val - target) <= tol:
                    break
            return guess
        def calc_perplexity(prob_matrix):
            """Calculate the perplexity of each row
            of a matrix of probabilities."""
            entropy = -np.sum(prob_matrix * np.log2(prob_matrix), 1)
            perplexity = 2 ** entropy
            return perplexity

        def perplexity(distances, sigmas, zero_index):
            """Wrapper function for quick calculation of
            perplexity over a distance matrix."""
            def calc_prob_matrix(distances, sigmas=None, zero_index=None):
                """Convert a distances matrix to a matrix of probabilities."""
                if sigmas is not None:
                    two_sig_sq = 2. * np.square(sigmas.reshape((-1, 1)))
                    return self.softmax(distances / two_sig_sq, zero_index=zero_index)
                else:
                    return self.softmax(distances, zero_index=zero_index)
            return calc_perplexity(
                calc_prob_matrix(distances, sigmas, zero_index))
        sigmas = []
        # For each row of the matrix (each point in our dataset)
        for i in range(self.distances.shape[0]):
            # Make fn that returns perplexity of this row given sigma
            eval_fn = lambda sigma: \
                perplexity(self.distances[i:i+1, :], np.array(sigma), i)
            # Binary search over sigmas to achieve target perplexity
            correct_sigma = binary_search(eval_fn, self.perplexity)
            # Append the resulting sigma to our output array
            sigmas.append(correct_sigma)
        return np.array(sigmas)


    def tsne_grad(self):
        """t-SNE: Estimate the gradient of the cost with respect to Y."""
        pq_diff = self.P - self.Q  # NxN matrix
        pq_expanded = np.expand_dims(pq_diff, 2)  # NxNx1
        y_diffs = np.expand_dims(self.Y, 1) - np.expand_dims(self.Y, 0)  # NxNx2
        # Expand our distances matrix so can multiply by y_diffs
        distances_expanded = np.expand_dims(self.distances, 2)  # NxNx1
        # Weight this (NxNx2) by distances matrix (NxNx1)
        y_diffs_wt = y_diffs * distances_expanded  # NxNx2
        grad = 4. * (pq_expanded * y_diffs_wt).sum(1)  # Nx2
        return grad

    def neg_squared_euc_dists(self, X):
        """Compute matrix containing negative squared euclidean
        distance for all pairs of points in input matrix X
        # Arguments:
            X: matrix of size NxD
        # Returns:
            NxN matrix D, with entry D_ij = negative squared
            euclidean distance between rows X_i and X_j
        """
        sum_X = np.sum(np.square(X), 1)
        D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
        return -D


    def softmax(self, X, diag_zero=True, zero_index=None):
        """Compute softmax values for each row of matrix X."""

        # Subtract max for numerical stability
        e_x = np.exp(X - np.max(X, axis=1).reshape([-1, 1]))

        # We usually want diagonal probailities to be 0.
        if zero_index is None:
            if diag_zero:
                np.fill_diagonal(e_x, 0.)
        else:
            e_x[:, zero_index] = 0.

        # Add a tiny constant for stability of log we take later
        e_x = e_x + 1e-8  # numerical stability

        return e_x / e_x.sum(axis=1).reshape([-1, 1])

    def q_tsne(self):
        """t-SNE: Given low-dimensional representations Y, compute
        matrix of joint probabilities with entries q_ij."""
        distances = self.neg_squared_euc_dists(self.Y)
        inv_distances = np.power(1. - distances, -1)
        np.fill_diagonal(inv_distances, 0.)
        return inv_distances / np.sum(inv_distances), inv_distances
    

class KMeans:
    def __init__(self, n_clusters, max_iters=1000, tol=1e-5):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.labels = None
        
    def fit(self, X):
        n_samples, n_features = X.shape
        
        # Initialize centroids randomly
        idx = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[idx]
        
        for i in range(self.max_iters):
            # Assign each data point to the nearest centroid
            distances = self._calc_distances(X)
            self.labels = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.zeros((self.n_clusters, n_features))
            for j in range(self.n_clusters):
                new_centroids[j] = np.mean(X[self.labels == j], axis=0)
                
            # Check for convergence
            if np.sum(np.abs(new_centroids - self.centroids)) < self.tol:
                break
                
            self.centroids = new_centroids
            
    def predict(self, X):
        distances = self._calc_distances(X)
        return np.argmin(distances, axis=1)
    
    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)
        
    def _calc_distances(self, X):
        distances = np.zeros((X.shape[0], self.n_clusters))
        for i, centroid in enumerate(self.centroids):
            distances[:, i] = np.linalg.norm(X - centroid, axis=1)
        return distances

def euclidean_distance(x1, x2):
    """ Calculates the l2 distance between two vectors """
    distance = 0
    # Squared distance between each coordinate
    for i in range(len(x1)):
        distance += pow((x1[i] - x2[i]), 2)
    return math.sqrt(distance)    

class KMedoids():
    """A simple clustering method that forms k clusters by first assigning
    samples to the closest medoids, and then swapping medoids with non-medoid
    samples if the total distance (cost) between the cluster members and their medoid
    is smaller than prevoisly.
    Parameters:
    -----------
    k: int
        The number of clusters the algorithm will form.
    """
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
        self.final_clusters = None
        self.centroids = None

    def _init_random_medoids(self, X):
        """ Initialize the medoids as random samples """
        n_samples, n_features = np.shape(X)
        medoids = np.zeros((self.n_clusters, n_features))
        for i in range(self.n_clusters):
            medoid = X[np.random.choice(range(n_samples))]
            medoids[i] = medoid
        return medoids

    def _closest_medoid(self, sample, medoids):
        """ Return the index of the closest medoid to the sample """
        closest_i = None
        closest_distance = float("inf")
        for i, medoid in enumerate(medoids):
            distance = euclidean_distance(sample, medoid)
            if distance < closest_distance:
                closest_i = i
                closest_distance = distance
        return closest_i

    def _create_clusters(self, X, medoids):
        """ Assign the samples to the closest medoids to create clusters """
        clusters = [[] for _ in range(self.n_clusters)]
        for sample_i, sample in enumerate(X):
            medoid_i = self._closest_medoid(sample, medoids)
            clusters[medoid_i].append(sample_i)
        return clusters

    def _calculate_cost(self, X, clusters, medoids):
        """ Calculate the cost (total distance between samples and their medoids) """
        cost = 0
        # For each cluster
        for i, cluster in enumerate(clusters):
            medoid = medoids[i]
            for sample_i in cluster:
                # Add distance between sample and medoid as cost
                cost += euclidean_distance(X[sample_i], medoid)
        return cost

    def _get_non_medoids(self, X, medoids):
        """ Returns a list of all samples that are not currently medoids """
        non_medoids = []
        for sample in X:
            if not sample in medoids:
                non_medoids.append(sample)
        return non_medoids

    def _get_cluster_labels(self, clusters, X):
        """ Classify samples as the index of their clusters """
        # One prediction for each sample
        y_pred = np.zeros(np.shape(X)[0])
        for cluster_i in range(len(clusters)):
            cluster = clusters[cluster_i]
            for sample_i in cluster:
                y_pred[sample_i] = cluster_i
        return y_pred
    
    def fit(self, X):
        """ Do Partitioning Around Medoids and return the cluster labels """
        # Initialize medoids randomly
        medoids = self._init_random_medoids(X)
        # Assign samples to closest medoids
        clusters = self._create_clusters(X, medoids)

        # Calculate the initial cost (total distance between samples and
        # corresponding medoids)
        cost = self._calculate_cost(X, clusters, medoids)

        # Iterate until we no longer have a cheaper cost
        while True:
            best_medoids = medoids
            lowest_cost = cost
            for medoid in medoids:
                # Get all non-medoid samples
                non_medoids = self._get_non_medoids(X, medoids)
                # Calculate the cost when swapping medoid and samples
                for sample in non_medoids:
                    # Swap sample with the medoid
                    new_medoids = medoids.copy()
                    new_medoids[medoids == medoid] = sample
                    # Assign samples to new medoids
                    new_clusters = self._create_clusters(X, new_medoids)
                    # Calculate the cost with the new set of medoids
                    new_cost = self._calculate_cost(
                        X, new_clusters, new_medoids)
                    # If the swap gives us a lower cost we save the medoids and cost
                    if new_cost < lowest_cost:
                        lowest_cost = new_cost
                        best_medoids = new_medoids
            # If there was a swap that resultet in a lower cost we save the
            # resulting medoids from the best swap and the new cost 
            if lowest_cost < cost:
                cost = lowest_cost
                medoids = best_medoids 
            # Else finished
            else:
                break

        self.final_clusters = self._create_clusters(X, medoids)
        self.centroids = medoids

    def predict(self, X):
        # Return the samples cluster indices as labels
        return self._get_cluster_labels(self.final_clusters, X)
    
    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)
    
def plot_silohuettes(range_n_clusters:Iterable, X, method):

    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        if method == 'KMeans':
            clusterer = KMeans(n_clusters=n_clusters)
        else:
            clusterer = KMedoids(n_clusters=n_clusters)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
        )

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(
            X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
        )

        # Labeling the clusters
        centers = clusterer.centroids
        # Draw white circles at cluster centers
        ax2.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k",
        )

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(
            f"Silhouette analysis for {method} clustering on sample data with n_clusters = {n_clusters}",
            fontsize=14,
            fontweight="bold",
        )

    plt.show()