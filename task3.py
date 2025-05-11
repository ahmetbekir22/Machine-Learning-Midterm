import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Function to load and preprocess a small sample of the MNIST dataset
def load_mnist_data():
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist.data.values, mnist.target
    X_sample, _, y_sample, _ = train_test_split(X, y, train_size=1000, stratify=y, random_state=42)
    return X_sample, y_sample

# Function to reduce the dimensionality of the data using PCA
def apply_pca(X_sample, n_components=3):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X_sample)


# Function to reconstruct the images using an MLPRegressor
def reconstruct_images_with_mlp(X_encoded, X_sample, epochs_list):
    reconstructed_images = {}
    for epochs in epochs_list:
        mlp = MLPRegressor(hidden_layer_sizes=(10, 50), max_iter=epochs, random_state=42)
        mlp.fit(X_encoded, X_sample)
        reconstructed_images[epochs] = mlp.predict(X_encoded)
    return reconstructed_images



# Function to plot original and reconstructed images side by side
def plot_images_with_epochs(original, reconstructions, epochs_list, num_images=10):
    for epochs in epochs_list:
        fig, axes = plt.subplots(2, num_images, figsize=(15, 6))
        fig.suptitle(f"Original VS Reconstructed Images (Epoch {epochs})", fontsize=16)

        for i in range(num_images):
            axes[0, i].imshow(original[i].reshape(28, 28), cmap='gray')
            axes[0, i].axis('off')
            axes[0, i].set_title("Original", fontsize=8)

        for i in range(num_images):
            axes[1, i].imshow(reconstructions[epochs][i].reshape(28, 28), cmap='gray')
            axes[1, i].axis('off')
            axes[1, i].set_title(f"Epoch {epochs}", fontsize=8)

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.show()

def main():
    X_sample, _ = load_mnist_data()
    X_encoded = apply_pca(X_sample, n_components=3)
    epochs_list = [10, 25, 500]
    reconstructed_images = reconstruct_images_with_mlp(X_encoded, X_sample, epochs_list)
    plot_images_with_epochs(X_sample[:10], reconstructed_images, epochs_list)

if __name__ == "__main__":
    main()
