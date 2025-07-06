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
    Converts organoid spike counts per bin into LSM input format.
    Args:
        organoid_spike_counts (list of lists): Data where each inner list contains the
                                             spike counts for the 8 electrodes in one time bin.
        num_lsm_inputs (int): Total number of input neurons in the LSM (e.g., 25).
        bin_size_s (float): The time window for counting spikes (in seconds).
        dt_ms (float): The simulation timestep in milliseconds.
        rng (Generator): NumPy random number generator.
    Returns:
        list: A list of NumPy arrays with spike times formatted for the LSM.
    """
    print(f"Converting spike counts to LSM input (bin size: {bin_size_s*1000}ms)...")
    lsm_spikes_list = [[] for _ in range(num_lsm_inputs)]
    neurons_per_electrode = num_lsm_inputs // 8
    remainder = num_lsm_inputs % 8
    for bin_index, counts_in_bin in enumerate(organoid_spike_counts):
        t_start = bin_index * bin_size_s
        t_stop = t_start + bin_size_s
        lsm_neuron_idx_start = 0
        for electrode_idx, spike_count in enumerate(counts_in_bin):
            num_assigned = neurons_per_electrode + (1 if electrode_idx < remainder else 0)
            if spike_count > 0:
                rate_hz = spike_count / bin_size_s
                generated_trains = generate_poisson_spike_trains(
                    n_neurons=num_assigned,
                    rate=rate_hz,
                    t_start=t_start,
                    t_stop=t_stop,
                    dt=dt_ms,
                    rng=rng
                )
                lsm_neuron_indices = range(lsm_neuron_idx_start, lsm_neuron_idx_start + num_assigned)
                for i, lsm_idx in enumerate(lsm_neuron_indices):
                    lsm_spikes_list[lsm_idx].extend(generated_trains[i])
            lsm_neuron_idx_start += num_assigned
    final_lsm_spikes = [np.array(spikes) for spikes in lsm_spikes_list]
    return final_lsm_spikes 