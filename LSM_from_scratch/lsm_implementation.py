import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d

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

    # Iterate through each time bin's data
    for bin_index, counts_in_bin in enumerate(organoid_spike_counts):
        t_start = bin_index * bin_size_s
        t_stop = t_start + bin_size_s
        
        lsm_neuron_idx_start = 0
        # Iterate through the 8 electrode counts in the current bin
        for electrode_idx, spike_count in enumerate(counts_in_bin):
            num_assigned = neurons_per_electrode + (1 if electrode_idx < remainder else 0)
            
            if spike_count > 0:
                rate_hz = spike_count / bin_size_s
                
                # Generate spikes for the assigned LSM neurons
                generated_trains = generate_poisson_spike_trains(
                    n_neurons=num_assigned,
                    rate=rate_hz,
                    t_start=t_start,
                    t_stop=t_stop,
                    dt=dt_ms,
                    rng=rng
                )
                
                # Add the generated spikes to the correct lists
                lsm_neuron_indices = range(lsm_neuron_idx_start, lsm_neuron_idx_start + num_assigned)
                for i, lsm_idx in enumerate(lsm_neuron_indices):
                    lsm_spikes_list[lsm_idx].extend(generated_trains[i])

            lsm_neuron_idx_start += num_assigned

    final_lsm_spikes = [np.array(spikes) for spikes in lsm_spikes_list]
    return final_lsm_spikes
# --- 1. Helper Function to Generate Input Spikes (Unchanged) ---
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

# --- 2. LIF Neuron Model Class (Unchanged) ---
class LIFNeuron:
    """Models a single Leaky Integrate-and-Fire (LIF) neuron."""
    def __init__(self, params, dt):
        self.dt = dt
        self.v_rest = params['v_rest']
        self.v_reset = params['v_reset']
        self.v_thresh = params['v_thresh']
        self.tau_m = params['tau_m']
        self.tau_refrac = params['tau_refrac']
        self.i_offset = params['i_offset']
        self.v = self.v_rest
        self.i_syn_E = 0.0
        self.i_syn_I = 0.0
        self.refractory_time = 0.0
        self.decay_m = np.exp(-dt / self.tau_m)
        self.decay_syn_E = np.exp(-dt / params['tau_syn_E'])
        self.decay_syn_I = np.exp(-dt / params['tau_syn_I'])

    def update(self):
        if self.refractory_time > 0:
            self.refractory_time -= self.dt
            self.v = self.v_reset
            self.i_syn_E *= self.decay_syn_E
            self.i_syn_I *= self.decay_syn_I
            return False

        self.i_syn_E *= self.decay_syn_E
        self.i_syn_I *= self.decay_syn_I
        total_current = self.i_offset + self.i_syn_E + self.i_syn_I
        self.v = self.v * self.decay_m + (1 - self.decay_m) * (self.v_rest + total_current * self.tau_m)
        
        if self.v >= self.v_thresh:
            self.v = self.v_reset
            self.refractory_time = self.tau_refrac
            return True
        return False

# --- 3. Network Class (Reverted to not include recurrent connections) ---
class Network:
    """Manages the entire simulation, including all neurons and connections."""
    def __init__(self, lif_params, n_input, n_exc, n_inh, dt=1.0, rng=None):
        self.dt = dt
        self.n_input = n_input
        self.n_exc = n_exc
        self.n_inh = n_inh
        self.n_total = n_exc + n_inh
        self.reservoir_neurons = [LIFNeuron(lif_params, dt) for _ in range(self.n_total)]
        self.input_connections = [[] for _ in range(n_input)]
        self.rng = rng if rng is not None else np.random.default_rng()

    def connect_inputs(self, w_input, delay_ms, p_connect=0.1):
        """Connects each input neuron to a random subset of reservoir neurons."""
        print(f"Connecting inputs with {p_connect*100}% probability...")
        delay_steps = int(delay_ms / self.dt)
        for i in range(self.n_input):
            for j in range(self.n_total):
                if self.rng.random() < p_connect:
                    self.input_connections[i].append((j, w_input, delay_steps))

    def run(self, duration_s, input_spike_trains):
        """Executes the simulation."""
        duration_ms = duration_s * 1000
        num_steps = int(duration_ms / self.dt)
        spike_queue = deque()
        self.spike_recorder = []
        
        print(f"Starting simulation for {duration_s}s ({num_steps} steps)...")
        
        for step in range(num_steps):
            current_time_ms = step * self.dt
            
            for neuron_idx, spike_times in enumerate(input_spike_trains):
                if np.any(np.isclose(spike_times, current_time_ms)):
                    for target_idx, weight, delay_steps in self.input_connections[neuron_idx]:
                        arrival_step = step + delay_steps
                        spike_queue.append((arrival_step, target_idx, weight))

            while spike_queue and spike_queue[0][0] == step:
                _, target_idx, weight = spike_queue.popleft()
                if weight > 0:
                    self.reservoir_neurons[target_idx].i_syn_E += weight
                else: 
                    self.reservoir_neurons[target_idx].i_syn_I += weight
            
            for i in range(self.n_total):
                if self.reservoir_neurons[i].update():
                    self.spike_recorder.append((current_time_ms, i))
        
        print("Simulation finished.")

def generate_mock_spike_counts(duration_s, bin_size_s, rng):
    """Creates mock spike count data with temporal structure."""
    print("Generating mock organoid spike count data...")
    num_bins = int(duration_s / bin_size_s)
    mock_counts = []
    
    # Create two distinct stimulation periods
    mid_point = num_bins // 2
    
    for bin_idx in range(num_bins):
        if bin_idx < mid_point:
            # First half: High activity period (5-15 spikes per electrode)
            counts_for_bin = rng.integers(5, 16, size=8).tolist()
        else:
            # Second half: Low activity period (0-5 spikes per electrode)
            counts_for_bin = rng.integers(0, 6, size=8).tolist()
        mock_counts.append(counts_for_bin)
    
    return mock_counts

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


if __name__ == '__main__':
    DT = 1.0  # ms
    TOTAL_DURATION = 20.0 # seconds
    
    LIF_PARAMS = {
        'v_rest': -65.0, 'v_reset': -65.0, 'v_thresh': -50.0,
        'tau_m': 20.0, 'tau_refrac': 5.0, 'tau_syn_E': 5.0,
        'tau_syn_I': 10.0, 'i_offset': 0.1,
    }

    N_INPUT, N_EXC, N_INH = 25, 160, 40
    N_TOTAL = N_EXC + N_INH
    N_HIDDEN_PLOT, N_OUTPUT_PLOT = 50, 25
    hidden_indices = list(range(N_HIDDEN_PLOT))
    output_indices = list(range(N_TOTAL - N_OUTPUT_PLOT, N_TOTAL))
    
    rng = np.random.default_rng(seed=42)
    
    # --- 1. Generate Mock Organoid Spike Count Data ---
    BIN_SIZE_S = 0.2  # 200ms bin size
    mock_organoid_counts = generate_mock_spike_counts(TOTAL_DURATION, BIN_SIZE_S, rng)

    # --- 2. Convert Spike Counts to LSM Input Format ---
    lsm_input_spikes = convert_spike_counts_to_lsm_input(
        organoid_spike_counts=mock_organoid_counts,
        num_lsm_inputs=N_INPUT,
        bin_size_s=BIN_SIZE_S,
        dt_ms=DT,
        rng=rng
    )

    # --- 3. Setup and Run the LSM ---
    lsm_network = Network(lif_params=LIF_PARAMS, n_input=N_INPUT, n_exc=N_EXC, n_inh=N_INH, dt=DT, rng=rng)
    lsm_network.connect_inputs(w_input=1.0, delay_ms=1.0, p_connect=0.1)
    
    # Pass the newly formatted spikes to the simulation
    lsm_network.run(duration_s=TOTAL_DURATION, input_spike_trains=lsm_input_spikes)

    # --- 4. Plot Results ---
    plot_activity_and_pca(lsm_network, TOTAL_DURATION, lsm_input_spikes, hidden_indices, output_indices)