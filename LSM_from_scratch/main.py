import numpy as np
from sklearn.decomposition import PCA # Keep PCA here for now for analysis
from scipy.ndimage import gaussian_filter1d # for smoothing
from lsm_core import Network, LIFNeuron 
from data_utils import generate_equation_based_data, convert_spike_counts_to_lsm_input, generate_poisson_spike_trains
from analysis import plot_activity_and_pca, train_and_evaluate_classifier

if __name__ == '__main__':
    # --- 1. Define Parameters ---
    DT = 1.0
    TOTAL_DURATION = 200.0  # Increased from 20.0 to 200.0 seconds
    BIN_SIZE_S = 0.2        # Keep original bin size
    
    LIF_PARAMS = { 'v_rest': -65.0, 'v_reset': -65.0, 'v_thresh': -50.0, 'tau_m': 20.0, 
                   'tau_refrac': 5.0, 'tau_syn_E': 5.0, 'tau_syn_I': 10.0, 'i_offset': 0.1 }

    N_INPUT, N_EXC, N_INH = 25, 160, 40
    N_RESERVOIR = N_EXC + N_INH
    
    rng = np.random.default_rng(seed=42)
    
    # --- 2. Generate Labeled Equation-Based Data ---
    half_duration = TOTAL_DURATION / 2
    
    # Generate data for the first half of the simulation using Pattern A ('sine')
    pattern_A_counts = generate_equation_based_data(half_duration, BIN_SIZE_S, 'sine', rng)
    # Generate data for the second half using Pattern B ('damped')
    pattern_B_counts = generate_equation_based_data(half_duration, BIN_SIZE_S, 'damped', rng)
    
    # Combine the two patterns into one continuous input stream
    combined_organoid_counts = pattern_A_counts + pattern_B_counts
    
    # Create corresponding labels for the classification task
    labels = np.array([1] * len(pattern_A_counts) + [0] * len(pattern_B_counts)) # 1 for Sine, 0 for Damped
    
    # --- 3. Convert Data and Run Simulation ---
    lsm_input_spikes = convert_spike_counts_to_lsm_input(
        organoid_spike_counts=combined_organoid_counts, num_lsm_inputs=N_INPUT,
        bin_size_s=BIN_SIZE_S, dt_ms=DT, rng=rng
    )
    
    lsm_network = Network(lif_params=LIF_PARAMS, n_input=N_INPUT, n_exc=N_EXC, n_inh=N_INH, dt=DT, rng=rng)
    lsm_network.connect_inputs(w_input=1.0, delay_ms=1.0, p_connect=0.15) # Slightly increased connectivity
    lsm_network.run(duration_s=TOTAL_DURATION, input_spike_trains=lsm_input_spikes)

    # --- 4. Analyze and Classify Results ---
    res_spikes = np.array(lsm_network.spike_recorder) if lsm_network.spike_recorder else np.empty((0,2))
    num_bins = len(combined_organoid_counts)
    analysis_bins = np.linspace(0, TOTAL_DURATION * 1000, num_bins + 1)
    
    reservoir_activity = np.array([
        np.histogram(res_spikes[res_spikes[:, 1] == i, 0], bins=analysis_bins)[0] 
        for i in range(N_RESERVOIR)
    ]).T
    
    smoothed_reservoir_activity = gaussian_filter1d(reservoir_activity, sigma=2, axis=0)

    train_and_evaluate_classifier(
        reservoir_activity=smoothed_reservoir_activity,
        labels=labels,
        n_components=5,
        class_names=['Pattern B (Damped)', 'Pattern A (Sine)'] # Update class names
    )