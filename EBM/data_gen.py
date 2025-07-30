import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration ---
RASTER_SHAPE = (8, 50)  # The dimensions for the FinalSpark data
N_SAMPLES_PER_CLASS = 2000 # Increased for a more robust dataset

def generate_baseline(shape, p=0.01):
    """Generates a raster with sparse, random noise."""
    return (np.random.rand(*shape) < p).astype(np.float32)

def generate_local_cluster(shape, patch_size=(4, 15), p_base=0.01, p_cluster=0.3):
    """Generates a raster with a single localized cluster of activity."""
    raster = generate_baseline(shape, p=p_base)
    y_start = np.random.randint(0, shape[0] - patch_size[0] + 1)
    x_start = np.random.randint(0, shape[1] - patch_size[1] + 1)
    cluster_patch = (np.random.rand(*patch_size) < p_cluster).astype(np.float32)
    raster[y_start:y_start+patch_size[0], x_start:x_start+patch_size[1]] = cluster_patch
    return raster

def generate_global_burst(shape, p=0.1):
    """Generates a raster with a higher global probability of firing."""
    return (np.random.rand(*shape) < p).astype(np.float32)

def generate_multi_cluster(shape, p_base=0.01, p_cluster=0.3):
    """Generates a raster with two distinct local clusters."""
    raster1 = generate_local_cluster(shape, patch_size=(3, 8), p_base=p_base, p_cluster=p_cluster)
    raster2 = generate_local_cluster(shape, patch_size=(3, 8), p_base=0, p_cluster=p_cluster)
    return np.clip(raster1 + raster2, 0, 1)

def generate_synthetic_dataset(n_samples_per_class):
    """Generates the full dataset with 4 classes of observations and labels."""
    observations = []
    labels = []
    class_generators = [
        generate_baseline, generate_local_cluster, 
        generate_global_burst, generate_multi_cluster
    ]
    for _ in range(n_samples_per_class):
        for class_idx, generator in enumerate(class_generators):
            observations.append(generator(RASTER_SHAPE))
            labels.append(class_idx)
    observations = np.array(observations)
    labels = np.array(labels)
    permutation = np.random.permutation(len(observations))
    return observations[permutation], labels[permutation]

def save_dataset(observations, labels, directory='data', filename='synthetic_dataset.npz'):
    """Saves the dataset to a compressed .npz file."""
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename)
    # Using savez_compressed is efficient for sparse binary data
    np.savez_compressed(filepath, observations=observations, labels=labels)
    print(f"✅ Dataset saved successfully to {filepath}")

# --- Main execution block ---
if __name__ == "__main__":
    total_samples = N_SAMPLES_PER_CLASS * 4
    print(f"Generating {total_samples} total samples...")
    
    observations, labels = generate_synthetic_dataset(N_SAMPLES_PER_CLASS)
    
    print("Dataset generation complete.")
    print(f"Observations shape: {observations.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # --- Save the generated dataset ---
    save_dataset(observations, labels)
    
    # --- Visualize one example from each class to verify ---
    print("\nVisualizing one sample from each class...")
    fig, axes = plt.subplots(1, 4, figsize=(16, 3))
    fig.suptitle("Example Generated Data from Each Class (8x50)", fontsize=16)
    
    class_names = ["Baseline", "Local Cluster", "Global Burst", "Multi-Cluster"]
    
    for i in range(4):
        idx = np.where(labels == i)[0][0]
        ax = axes[i]
        ax.imshow(observations[idx], cmap='gray_r', interpolation='nearest', aspect='auto')
        ax.set_title(f'Class {i}: {class_names[i]}')
        ax.set_xticks([]); ax.set_yticks([])
        
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()
