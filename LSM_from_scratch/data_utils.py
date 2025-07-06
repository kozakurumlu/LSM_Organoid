import numpy as np

def generate_poisson_spike_trains(n_neurons, rate, t_start, t_stop, dt=1.0, rng=None):
    """Generates a list of spike time arrays for a population of neurons."""
    if rng is None:
        rng = np.random.default_rng()
    spike_trains = []
    rate_per_ms = rate / 1000.0
    for _ in range(n_neurons):
        num_steps = int((t_stop - t_start) * 1000 / dt)
        spikes_at_step = rng.random(num_steps) < rate_per_ms * dt
        spike_indices = np.where(spikes_at_step)[0]
        neuron_spikes = (t_start * 1000) + spike_indices * dt
        spike_trains.append(neuron_spikes)
    return spike_trains

def generate_mock_spike_counts(duration_s, bin_size_s, rng):
    """Creates mock spike count data with temporal structure."""
    print("Generating mock organoid spike count data...")
    num_bins = int(duration_s / bin_size_s)
    mock_counts = []
    mid_point = num_bins // 2
    for bin_idx in range(num_bins):
        if bin_idx < mid_point:
            counts_for_bin = rng.integers(5, 16, size=8).tolist()
        else:
            counts_for_bin = rng.integers(0, 6, size=8).tolist()
        mock_counts.append(counts_for_bin)
    return mock_counts


def convert_spike_counts_to_lsm_input(organoid_spike_counts, num_lsm_inputs, bin_size_s, dt_ms, rng):
    """
    Converts organoid spike counts into LSM input spikes, with each of the 8
    data channels mapped to a dedicated group of LSM input neurons.
    """
    print(f"Converting spike counts to LSM input (spatially separate)...")
    
    if num_lsm_inputs % 8 != 0:
        raise ValueError("num_lsm_inputs must be divisible by 8 for this mapping.")
        
    lsm_spikes_list = [[] for _ in range(num_lsm_inputs)]
    neurons_per_channel = num_lsm_inputs // 8

    # Iterate through each time bin's data
    for bin_index, counts_in_bin in enumerate(organoid_spike_counts):
        t_start = bin_index * bin_size_s
        t_stop = t_start + bin_size_s
        
        # Iterate through the 8 electrode counts in the current bin
        for electrode_idx, spike_count in enumerate(counts_in_bin):
            if spike_count > 0:
                rate_hz = spike_count / bin_size_s
                
                # Determine the dedicated group of LSM neurons for this electrode
                start_neuron_idx = electrode_idx * neurons_per_channel
                end_neuron_idx = start_neuron_idx + neurons_per_channel
                lsm_neuron_indices = range(start_neuron_idx, end_neuron_idx)
                
                # Generate spikes for this specific group
                generated_trains = generate_poisson_spike_trains(
                    n_neurons=neurons_per_channel,
                    rate=rate_hz,
                    t_start=t_start,
                    t_stop=t_stop,
                    dt=dt_ms,
                    rng=rng
                )
                
                # Add the generated spikes to the correct lists
                for i, lsm_idx in enumerate(lsm_neuron_indices):
                    lsm_spikes_list[lsm_idx].extend(generated_trains[i])

    final_lsm_spikes = [np.array(spikes, dtype=float) for spikes in lsm_spikes_list]
    return final_lsm_spikes

def generate_equation_based_data(duration_s, bin_size_s, pattern_type='sine', noise_level=0, rng=None):
    """
    Generates spike count data based on one of two mathematical patterns.

    Args:
        duration_s (float): The duration of the pattern in seconds.
        bin_size_s (float): The time window for each spike count.
        pattern_type (str): Either 'sine' or 'random_walk'.
        noise_level (int): The amplitude of noise to add.
        rng (np.random.Generator): NumPy random number generator.

    Returns:
        list of lists: The generated spike count data.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    print(f"Generating data for pattern: '{pattern_type}'...")
    num_bins = int(duration_s / bin_size_s)
    t = np.linspace(0, duration_s, num_bins)
    
    equation_data = np.zeros((num_bins, 8))
    
    if pattern_type == 'sine':
        # Pattern A: A steady, periodic sine wave
        base_freq = 0.5 # Hz
        for i in range(8):
            freq = base_freq + rng.uniform(-0.1, 0.1)
            phase = rng.uniform(0, np.pi)
            wave = 10 * np.sin(2 * np.pi * freq * t + phase) + 10
            equation_data[:, i] = wave
            
    elif pattern_type == 'random_walk':
        # Pattern B: A meandering random walk for each channel
        # Start each channel at a different baseline count
        current_values = rng.uniform(5, 15, size=8)
        for t_step in range(num_bins):
            equation_data[t_step, :] = current_values
            # Update each channel's value with a small random step
            steps = rng.choice([-1, 0, 1], size=8)
            current_values += steps
            # Ensure counts don't go below zero
            current_values = np.maximum(0, current_values)
            
    # Add noise to the generated wave before converting to integer counts
    noise = rng.integers(-noise_level, noise_level + 1, size=equation_data.shape)
    noisy_data = equation_data + noise
    
    # Final conversion to integer counts, ensuring no negatives
    return np.maximum(0, noisy_data).astype(int).tolist()