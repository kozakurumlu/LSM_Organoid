import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
RASTER_SHAPE = (64, 64)  # The dimensions of our synthetic raster
N_SAMPLES_PER_CLASS = 1 # Number of examples to generate for each class

def generate_baseline(shape, p=0.01):
    """Generates a raster with sparse, random noise."""
    return (np.random.rand(*shape) < p).astype(np.float32)

def generate_local_cluster(shape, patch_size=10, p_base=0.01, p_cluster=0.3):
    """Generates a raster with a single localized cluster of activity."""
    # Start with a baseline raster
    raster = generate_baseline(shape, p=p_base)
    
    # Randomly select a top-left corner for the patch
    x_start = np.random.randint(0, shape[1] - patch_size)
    y_start = np.random.randint(0, shape[0] - patch_size)
    
    # Create the cluster within the patch
    cluster_patch = (np.random.rand(patch_size, patch_size) < p_cluster).astype(np.float32)
    raster[y_start:y_start+patch_size, x_start:x_start+patch_size] = cluster_patch
    
    return raster

def generate_global_burst(shape, p=0.1):
    """Generates a raster with a higher global probability of firing."""
    return (np.random.rand(*shape) < p).astype(np.float32)

def generate_synthetic_dataset(n_samples_per_class):
    """Generates the full dataset with observations and labels."""
    observations = []
    labels = []
    
    for i in range(n_samples_per_class):
        # Class 0: Baseline
        observations.append(generate_baseline(RASTER_SHAPE))
        labels.append(0)
        
        # Class 1: Local Cluster
        observations.append(generate_local_cluster(RASTER_SHAPE))
        labels.append(1)
        
        # Class 2: Global Burst
        observations.append(generate_global_burst(RASTER_SHAPE))
        labels.append(2)
        
    # Convert lists to numpy arrays and shuffle them together
    observations = np.array(observations)
    labels = np.array(labels)
    
    permutation = np.random.permutation(len(observations))
    observations = observations[permutation]
    labels = labels[permutation]
    
    return observations, labels

# --- Main execution block ---
if __name__ == "__main__":
    print(f"Generating {N_SAMPLES_PER_CLASS * 3} total samples...")
    
    # Generate the dataset
    observations, labels = generate_synthetic_dataset(N_SAMPLES_PER_CLASS)
    
    print("Dataset generated successfully.")
    print(f"Observations shape: {observations.shape}") # Should be (3000, 32, 32)
    print(f"Labels shape: {labels.shape}")       # Should be (3000,)
    
    # --- Visualize one example from each class to verify ---
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("Example Generated Data from Each Class", fontsize=16)
    
    for i in range(3):
        # Find the first occurrence of each class label
        idx = np.where(labels == i)[0][0]
        ax = axes[i]
        ax.imshow(observations[idx], cmap='gray_r', interpolation='nearest')
        ax.set_title(f'Class {i}: {"Baseline" if i==0 else "Local Cluster" if i==1 else "Global Burst"}')
        ax.set_xticks([])
        ax.set_yticks([])
        
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()