import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def plot_scree(pca):
    """
    Plots the explained variance of each principal component.

    Args:
        pca (PCA): A fitted PCA object from scikit-learn.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, 'o-', linewidth=2)
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')
    plt.grid(True)
    plt.show()


def plot_activity_and_pca(network, total_duration_s, input_spikes, hidden_indices, output_indices):
    """
    Generates raster plots and performs a single PCA on all populations,
    displaying the resulting trajectories on one 3D graph.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    # --- Part 1: Raster Plots ---
    fig_raster, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True, gridspec_kw={'hspace': 0.3})
    fig_raster.suptitle('LSM Activity by Population', fontsize=16)
    input_spikes_t, input_spikes_id = [], []
    for i, spikes in enumerate(input_spikes):
        input_spikes_t.extend(spikes)
        input_spikes_id.extend([i] * len(spikes))
    ax1.scatter(np.array(input_spikes_t) / 1000.0, input_spikes_id, c='blue', marker='.', s=10)
    ax1.set_title('Input Layer Activity')
    ax1.set_ylabel('Neuron Index')
    ax1.set_ylim(-1, network.n_input)
    res_spikes = np.array(network.spike_recorder) if network.spike_recorder else np.empty((0,2))
    hidden_mask = np.isin(res_spikes[:, 1], hidden_indices)
    ax2.scatter(res_spikes[hidden_mask, 0] / 1000.0, res_spikes[hidden_mask, 1], c='green', marker='.', s=10)
    ax2.set_title('Hidden Reservoir Activity')
    ax2.set_ylabel('Neuron Index')
    if hidden_indices: ax2.set_ylim(min(hidden_indices)-1, max(hidden_indices)+1)
    output_mask = np.isin(res_spikes[:, 1], output_indices)
    ax3.scatter(res_spikes[output_mask, 0] / 1000.0, res_spikes[output_mask, 1], c='purple', marker='.', s=10)
    ax3.set_title('Output Reservoir Activity')
    ax3.set_ylabel('Neuron Index')
    ax3.set_xlabel('Time (s)')
    if output_indices: ax3.set_ylim(min(output_indices)-1, max(output_indices)+1)
    for ax in [ax1, ax2, ax3]:
        ax.axvline(total_duration_s / 2, color='gray', linestyle='--')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    # --- Part 2: PCA on a Shared Coordinate System ---
    print("\nPerforming PCA on a shared coordinate system...")
    bin_size_ms = 20.0
    num_bins = int(total_duration_s * 1000 / bin_size_ms)
    time_bins = np.linspace(0, total_duration_s * 1000, num_bins + 1)
    activity_input = np.array([np.histogram(s, bins=time_bins)[0] for s in input_spikes]).T
    activity_hidden = np.array([np.histogram(res_spikes[res_spikes[:, 1] == i, 0], bins=time_bins)[0] for i in hidden_indices]).T
    activity_output = np.array([np.histogram(res_spikes[res_spikes[:, 1] == i, 0], bins=time_bins)[0] for i in output_indices]).T
    sigma_smooth = 2.0
    smooth_input = gaussian_filter1d(activity_input, sigma=sigma_smooth, axis=0)
    smooth_hidden = gaussian_filter1d(activity_hidden, sigma=sigma_smooth, axis=0)
    smooth_output = gaussian_filter1d(activity_output, sigma=sigma_smooth, axis=0)
    combined_activity = np.hstack([smooth_input, smooth_hidden, smooth_output])
    pca = PCA(n_components=3)
    pca.fit(combined_activity)
    pad_hidden = np.zeros_like(smooth_hidden)
    pad_output = np.zeros_like(smooth_output)
    padded_input_activity = np.hstack([smooth_input, pad_hidden, pad_output])
    pad_input = np.zeros_like(smooth_input)
    padded_hidden_activity = np.hstack([pad_input, smooth_hidden, pad_output])
    padded_output_activity = np.hstack([pad_input, pad_hidden, smooth_output])
    pc_input = pca.transform(padded_input_activity)
    pc_hidden = pca.transform(padded_hidden_activity)
    pc_output = pca.transform(padded_output_activity)
    fig_pca = plt.figure(figsize=(12, 10))
    ax = fig_pca.add_subplot(111, projection='3d')
    split_point = num_bins // 2
    colors = {'high': ['#ff7b7b', '#ff0000', '#8b0000'], 'low': ['#7b7bff', '#0000ff', '#00008b']}
    labels = ['Input', 'Hidden', 'Output']
    pcas = [pc_input, pc_hidden, pc_output]
    for i, (pc_data, label) in enumerate(zip(pcas, labels)):
        ax.plot(pc_data[:split_point, 0], pc_data[:split_point, 1], pc_data[:split_point, 2], 
                label=f'{label} (High Activity)', c=colors['high'][i], alpha=0.9)
        ax.plot(pc_data[split_point:, 0], pc_data[split_point:, 1], pc_data[split_point:, 2], 
                label=f'{label} (Low Activity)', c=colors['low'][i], alpha=0.9)
    ax.set_title('3D PCA of State Trajectories in a Shared Space')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.legend()
    plt.show()

def plot_scree(pca, ax=None):
    """Plot a scree plot for PCA explained variance."""
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_, marker='o')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_title('Scree Plot')
    plt.show()

def plot_reservoir_activity(spike_recorder, duration_s, n_total):
    """
    Creates a raster plot of the reservoir's spiking activity.
    """
    if not spike_recorder:
        print("Reservoir recorded no spikes.")
        return

    spike_times, neuron_ids = zip(*spike_recorder)
    
    plt.figure(figsize=(15, 6))
    plt.scatter(np.array(spike_times) / 1000.0, neuron_ids, marker='.', s=5, c='k', alpha=0.6)
    plt.axvline(duration_s / 2, color='r', linestyle='--', label='Pattern Switch')
    plt.title('Reservoir Spike Activity Raster Plot')
    plt.xlabel('Time (s)')
    plt.ylabel('Reservoir Neuron Index')
    plt.ylim(-1, n_total)
    plt.xlim(0, duration_s)
    plt.legend()
    plt.grid(True)
    plt.show()

def train_and_evaluate_classifier(reservoir_activity, labels, n_components=5, class_names=None):
    """
    Trains and evaluates a logistic regression classifier on reservoir states.

    Args:
        reservoir_activity (np.ndarray): The smoothed activity of reservoir neurons (time, neurons).
        labels (np.ndarray): The corresponding label (0 or 1) for each time step.
        n_components (int): The number of principal components to use as features.
        class_names (list or None): List of class names for reporting. If None, defaults to ['Low Activity', 'High Activity'].
    """
    print(f"\n--- Training Classifier using {n_components} Principal Components ---")
    if class_names is None:
        class_names = ['Low Activity', 'High Activity']
    # 1. Use PCA for dimensionality reduction
    pca = PCA(n_components=n_components)
    features = pca.fit_transform(reservoir_activity)
    # Optional: Plot the scree plot to validate n_components
    plot_scree(pca)
    # 2. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=42, stratify=labels
    )
    # 3. Train a logistic regression model
    classifier = LogisticRegression(random_state=42)
    classifier.fit(X_train, y_train)
    # 4. Evaluate the model
    accuracy = classifier.score(X_test, y_test)
    print(f"Classifier Accuracy: {accuracy:.2%}")
    # Print a detailed classification report
    predictions = classifier.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=class_names))
    # Display a confusion matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot()
    plt.show()