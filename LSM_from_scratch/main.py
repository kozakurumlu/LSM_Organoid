import numpy as np
from tqdm import tqdm
import os
from lsm_core import Network, LIFNeuron
from data_utils import generate_equation_based_data, convert_spike_counts_to_lsm_input, generate_poisson_spike_trains
from analysis import plot_activity_and_pca, train_and_evaluate_classifier, gaussian_filter1d

if __name__ == '__main__':
    # --- 1. Define Parameters & File Paths ---
    DT = 1.0
    TRIAL_DURATION = 20.0
    NUM_TRIALS = 100  # Running 100 trials as you suggested
    BIN_SIZE_S = 0.2
    
    LIF_PARAMS = { 'v_rest': -65.0, 'v_reset': -65.0, 'v_thresh': -50.0, 'tau_m': 20.0, 
                   'tau_refrac': 5.0, 'tau_syn_E': 5.0, 'tau_syn_I': 10.0, 'i_offset': 0.15 }

    N_INPUT, N_EXC, N_INH = 25, 160, 40
    N_RESERVOIR = N_EXC + N_INH
    
    rng = np.random.default_rng(seed=42)
    
    # Define file paths for saved data
    RESERVOIR_DATA_PATH = 'reservoir_states.npy'
    INPUT_DATA_PATH = 'input_counts.npy'
    LABELS_PATH = 'labels.npy'

    # --- 2. Load Data or Run Simulation ---
    if os.path.exists(RESERVOIR_DATA_PATH):
        print("Loading saved simulation data from disk...")
        final_reservoir_states = np.load(RESERVOIR_DATA_PATH)
        final_input_counts = np.load(INPUT_DATA_PATH)
        final_labels = np.load(LABELS_PATH)
    else:
        print("No saved data found. Running full simulation...")
        all_reservoir_states = []
        all_input_counts = []
        all_labels = []

        print(f"Running {NUM_TRIALS} trials of {TRIAL_DURATION}s each...")
        for _ in tqdm(range(NUM_TRIALS)):
            # --- Generate data for this trial ---
            half_duration = TRIAL_DURATION / 2
            pattern_A_counts = generate_equation_based_data(half_duration, BIN_SIZE_S, 'sine', noise_level=3, rng=rng)
            pattern_B_counts = generate_equation_based_data(half_duration, BIN_SIZE_S, 'random_walk', noise_level=3, rng=rng)
            
            combined_counts = pattern_A_counts + pattern_B_counts
            labels = np.array([1] * len(pattern_A_counts) + [0] * len(pattern_B_counts))
            
            # --- Run simulation for this trial ---
            lsm_input_spikes = convert_spike_counts_to_lsm_input(
                organoid_spike_counts=combined_counts, num_lsm_inputs=N_INPUT,
                bin_size_s=BIN_SIZE_S, dt_ms=DT, rng=rng
            )
            
            lsm_network = Network(lif_params=LIF_PARAMS, n_input=N_INPUT, n_exc=N_EXC, n_inh=N_INH, dt=DT, rng=rng)
            lsm_network.connect_inputs(w_input=1.0, delay_ms=1.0, p_connect=0.2)

            # Define a dictionary of weights for the internal connections
            reservoir_weights = {
                'ee': 0.2,   # Excitatory -> Excitatory
                'ei': 0.5,   # Excitatory -> Inhibitory
                'ie': 0.4,   # Inhibitory -> Excitatory
                'ii': 0.4    # Inhibitory -> Inhibitory
            }
            # Connect the reservoir internally with the new method
            lsm_network.connect_reservoir(
                weights=reservoir_weights,
                delay_ms=1.0, 
                p_connect=0.1
            )

            lsm_network.run(duration_s=TRIAL_DURATION, input_spike_trains=lsm_input_spikes)

            # --- Collect and process results for this trial ---
            res_spikes = np.array(lsm_network.spike_recorder) if lsm_network.spike_recorder else np.empty((0,2))
            num_bins = len(combined_counts)
            analysis_bins = np.linspace(0, TRIAL_DURATION * 1000, num_bins + 1)
        
            reservoir_activity = np.array([
                np.histogram(res_spikes[res_spikes[:, 1] == i, 0], bins=analysis_bins)[0] 
                for i in range(N_RESERVOIR)
            ]).T
            
            all_reservoir_states.append(reservoir_activity)
            all_input_counts.append(combined_counts)
            all_labels.append(labels)

        # --- Combine and Save Data From All Trials ---
        final_reservoir_states = np.vstack(all_reservoir_states)
        final_input_counts = np.vstack(all_input_counts)
        final_labels = np.concatenate(all_labels)

        print("Saving generated data to disk...")
        np.save(RESERVOIR_DATA_PATH, final_reservoir_states)
        np.save(INPUT_DATA_PATH, final_input_counts)
        np.save(LABELS_PATH, final_labels)

    # --- 3. Process Data for Classification ---
    smoothed_reservoir_activity = gaussian_filter1d(final_reservoir_states, sigma=2, axis=0)

    # --- 4. Train and Evaluate Classifiers ---
    
    # A) Baseline Classifier (on raw, noisy input)
    print("\n" + "="*50)
    print("  Training Baseline Classifier (on raw noisy input data)")
    print("="*50)
    train_and_evaluate_classifier(
        reservoir_activity=final_input_counts,
        labels=final_labels,
        n_components=7,
        class_names=['Pattern B (Random Walk)', 'Pattern A (Sine)']
    )
    
    # B) LSM Classifier (on reservoir states)
    print("\n" + "="*50)
    print("  Training LSM Classifier (on processed reservoir states)")
    print("="*50)
    train_and_evaluate_classifier(
        reservoir_activity=smoothed_reservoir_activity,
        labels=final_labels,
        n_components=5,
        class_names=['Pattern B (Random Walk)', 'Pattern A (Sine)']
    )