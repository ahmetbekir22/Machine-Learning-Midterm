import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import time

# 1. Data Generation (5 Features, Simpler Clusters)
X, y = make_classification(n_samples=3000, n_features=5, n_informative=3, 
                           n_redundant=0, n_classes=5, n_clusters_per_class=1, random_state=42)

# 2. Data Splitting and Normalization
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. K-Means Clustering (20 Clusters)
kmeans = KMeans(n_clusters=20, random_state=42)
clusters = kmeans.fit_predict(X_train)
centroids = kmeans.cluster_centers_

# 4. Density-Based Cluster Selection
densities = {c: np.var(X_train[clusters == c]) for c in np.unique(clusters)}
selected_clusters = sorted(densities, key=densities.get)[:10]

# 5. Single-Stage Sampling
single_stage_idx = np.isin(clusters, selected_clusters)
X_single_stage = X_train[single_stage_idx]
y_single_stage = y_train[single_stage_idx]

# 6. Double-Stage Sampling
def double_stage_sampling(X, labels, selected_clusters, samples_per_cluster=30, random_state=42):
    np.random.seed(random_state)
    sampled_indices = []
    for cluster in selected_clusters:
        cluster_points = np.where(labels == cluster)[0]
        sampled_indices.extend(np.random.choice(cluster_points, size=samples_per_cluster, replace=False))
    return sampled_indices

double_stage_indices = double_stage_sampling(X_train, clusters, selected_clusters)
X_double_stage = X_train[double_stage_indices]
y_double_stage = y_train[double_stage_indices]

# 7. Training and Evaluating the MLP Model
def train_evaluate_mlp(X_train, y_train, X_test, y_test):
    mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=600, 
                        learning_rate_init=0.001, learning_rate='adaptive', random_state=42)
    start_time = time.time()
    mlp.fit(X_train, y_train)
    training_time = time.time() - start_time
    accuracy = accuracy_score(y_test, mlp.predict(X_test))
    return accuracy, training_time

# 8. Testing 
orig_acc, orig_time = train_evaluate_mlp(X_train, y_train, X_test, y_test)
single_acc, single_time = train_evaluate_mlp(X_single_stage, y_single_stage, X_test, y_test)
double_acc, double_time = train_evaluate_mlp(X_double_stage, y_double_stage, X_test, y_test)

print("\nResults:")
print(f"(Original Data) Mean Testing Accuracy: {orig_acc:.3f} Training Time: {orig_time*1000:.3f} ms")
print(f"(Single-stage Clustering) Mean Testing Accuracy: {single_acc:.3f} Training Time: {single_time*1000:.3f} ms")
print(f"(Double-stage Clustering) Mean Testing Accuracy: {double_acc:.3f} Training Time: {double_time*1000:.3f} ms")

# 10. Visualization
plt.figure(figsize=(12, 12))

# Original Data
plt.subplot(2, 2, 1)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', edgecolor='k')
plt.title("Original Training Data")

# Data After Clustering
plt.subplot(2, 2, 2)
plt.scatter(X_train[:, 0], X_train[:, 1], c=clusters, cmap='tab20', edgecolor='k')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, marker='X', label='Centroids')
plt.title("Clustered Training Data")
plt.legend()

# Single-Stage Sampling
plt.subplot(2, 2, 3)
plt.scatter(X_single_stage[:, 0], X_single_stage[:, 1], c=y_single_stage, cmap='tab20', edgecolor='k')
plt.title("Single-Stage Sampling")

# Double-Stage Sampling
plt.subplot(2, 2, 4)
plt.scatter(X_double_stage[:, 0], X_double_stage[:, 1], c=y_double_stage, cmap='tab20', edgecolor='k')
plt.title("Double-Stage Sampling")

plt.tight_layout()
plt.show()
