import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# --- 1. Helper Function to Generate Input Spikes ---
# This function is kept from the original example to create our input data.
def generate_poisson_spike_trains(n_neurons, rate, t_start, t_stop, dt=1.0, rng=None):
    """
    Generates a list of spike time arrays for a population of neurons
    firing with a given Poisson rate.

    Args:
        n_neurons (int): Number of neurons in the population.
        rate (float): Firing rate in Hz.
        t_start (float): Start time in seconds.
        t_stop (float): Stop time in seconds.
        dt (float): Timestep in milliseconds.
        rng (Generator, optional): NumPy random number generator. Defaults to None.

    Returns:
        list: A list of NumPy arrays, where each array contains the spike times (in ms)
              for a single neuron.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    spike_trains = []
    # Convert rate from Hz (spikes/sec) to spikes/ms
    rate_per_ms = rate / 1000.0
    
    for _ in range(n_neurons):
        # Generate spike times for one neuron
        neuron_spikes = []
        # Calculate number of time steps
        num_steps = int((t_stop - t_start) * 1000 / dt)
        # For each time step, draw from a Bernoulli distribution (prob = rate_per_ms * dt)
        # This is an efficient way to simulate a Poisson process in discrete time.
        spikes_at_step = rng.random(num_steps) < rate_per_ms * dt
        
        # Get the indices (time steps) where spikes occurred
        spike_indices = np.where(spikes_at_step)[0]
        
        # Convert time steps to simulation time in ms
        neuron_spikes = (t_start * 1000) + spike_indices * dt
        spike_trains.append(neuron_spikes)
        
    return spike_trains

# --- 2. LIF Neuron Model Class ---
class LIFNeuron:
    """
    Models a single Leaky Integrate-and-Fire (LIF) neuron.
    The dynamics are solved using the Euler method.
    """
    def __init__(self, params, dt):
        """
        Initializes the neuron with its specific parameters.

        Args:
            params (dict): Dictionary containing neuron parameters.
            dt (float): Simulation timestep in ms.
        """
        self.dt = dt
        # Neuron parameters
        self.v_rest = params['v_rest']
        self.v_reset = params['v_reset']
        self.v_thresh = params['v_thresh']
        self.tau_m = params['tau_m']
        self.tau_refrac = params['tau_refrac']
        self.tau_syn_E = params['tau_syn_E']
        self.tau_syn_I = params['tau_syn_I']
        self.i_offset = params['i_offset']

        # State variables
        self.v = self.v_rest  # Membrane potential, starts at rest
        self.i_syn_E = 0.0     # Excitatory synaptic current
        self.i_syn_I = 0.0     # Inhibitory synaptic current
        self.refractory_time = 0.0 # Time left in refractory period

        # Pre-calculate decay constants for efficiency
        self.decay_m = np.exp(-dt / self.tau_m)
        self.decay_syn_E = np.exp(-dt / self.tau_syn_E)
        self.decay_syn_I = np.exp(-dt / self.tau_syn_I)

    def update(self):
        """
        Updates the neuron's state for one time step.
        """
        # If in refractory period, do nothing but decrement the timer
        if self.refractory_time > 0:
            self.refractory_time -= self.dt
            # Ensure voltage stays at reset potential during refractory period
            self.v = self.v_reset
            # Currents still decay
            self.i_syn_E *= self.decay_syn_E
            self.i_syn_I *= self.decay_syn_I
            return False # Did not spike

        # --- Update Synaptic Currents ---
        # Currents decay exponentially
        self.i_syn_E *= self.decay_syn_E
        self.i_syn_I *= self.decay_syn_I
        
        # --- Update Membrane Potential (Euler integration) ---
        # This is the discretized version of the LIF differential equation:
        # tau_m * dV/dt = -(V - V_rest) + I_offset + I_syn
        total_current = self.i_offset + self.i_syn_E + self.i_syn_I
        
        # More stable integration method than direct Euler
        self.v = self.v * self.decay_m + (1 - self.decay_m) * (self.v_rest + total_current * self.tau_m)
        
        # --- Check for Spike ---
        if self.v >= self.v_thresh:
            self.v = self.v_reset  # Reset potential
            self.refractory_time = self.tau_refrac # Start refractory period
            return True # Did spike
            
        return False # Did not spike

# --- 3. Network Class ---
class Network:
    """
    Manages the entire simulation, including all neurons, connections,
    and the main simulation loop.
    """
    def __init__(self, lif_params, n_input, n_exc, n_inh, dt=1.0):
        """
        Initializes the network.
        
        Args:
            lif_params (dict): Parameters for the LIF neurons.
            n_input (int): Number of input neurons.
            n_exc (int): Number of excitatory neurons in the reservoir.
            n_inh (int): Number of inhibitory neurons in the reservoir.
            dt (float): Simulation timestep in ms.
        """
        self.dt = dt
        self.n_input = n_input
        self.n_exc = n_exc
        self.n_inh = n_inh
        self.n_total = n_exc + n_inh

        # Create neuron populations
        self.reservoir_neurons = [LIFNeuron(lif_params, dt) for _ in range(self.n_total)]
        
        # --- Connectivity ---
        # Adjacency list: self.connections[i] contains a list of targets for neuron i
        # Each target is a tuple: (target_idx, weight, delay_steps)
        self.input_connections = [[] for _ in range(n_input)]
        self.recurrent_connections = [[] for _ in range(self.n_total)]

        # --- Data Recorders ---
        self.spike_recorder = [] # List of (time_ms, neuron_idx) tuples
        self.voltage_recorder = [] # List to store voltage traces

    def connect_inputs(self, w_input, delay_ms):
        """
        Connects input neurons to specific reservoir neurons.
        In this example, we connect one-to-one for simplicity.
        """
        delay_steps = int(delay_ms / self.dt)
        # Connect input_high_rate to first n_input neurons
        for i in range(self.n_input):
            # The weight is positive, so it's excitatory
            self.input_connections[i].append((i, w_input, delay_steps))
        # Connect input_low_rate to the next n_input neurons
        for i in range(self.n_input):
            # Target neuron index is shifted by n_input
            self.input_connections[i].append((i + self.n_input, w_input, delay_steps))

    def run(self, duration_s, input_spike_trains_high, input_spike_trains_low, neurons_to_record_v):
        """
        Executes the main simulation loop.
        
        Args:
            duration_s (float): Total simulation time in seconds.
            input_spike_trains_high (list): Spikes for the high-rate input.
            input_spike_trains_low (list): Spikes for the low-rate input.
            neurons_to_record_v (list): Indices of neurons whose voltage we want to record.
        """
        duration_ms = duration_s * 1000
        num_steps = int(duration_ms / self.dt)
        
        # The "spike queue" handles synaptic delays. It's a deque for efficient appends and pops.
        # Each item is a tuple: (arrival_step, target_neuron_idx, weight)
        spike_queue = deque()

        # Initialize voltage recorder with empty lists for each neuron to be recorded
        self.voltage_recorder = [[] for _ in neurons_to_record_v]
        
        print(f"Starting simulation for {duration_s}s ({num_steps} steps)...")
        
        # --- Main Simulation Loop ---
        for step in range(num_steps):
            current_time_ms = step * self.dt

            # --- 1. Process External Input Spikes ---
            # Check for spikes from the high-rate input group
            for neuron_idx, spike_times in enumerate(input_spike_trains_high):
                if current_time_ms in spike_times:
                    # Propagate this spike to its connected reservoir neurons
                    for target_idx, weight, delay_steps in self.input_connections[neuron_idx]:
                        arrival_step = step + delay_steps
                        spike_queue.append((arrival_step, target_idx, weight))

            # Check for spikes from the low-rate input group
            for neuron_idx, spike_times in enumerate(input_spike_trains_low):
                if current_time_ms in spike_times:
                    # Propagate this spike
                    for target_idx, weight, delay_steps in self.input_connections[neuron_idx]:
                        # The target index is different for this group
                        if target_idx >= self.n_input: 
                            arrival_step = step + delay_steps
                            spike_queue.append((arrival_step, target_idx, weight))
            
            # --- 2. Process Arriving Spikes from Queue ---
            # Check for any spikes (recurrent or input) that are scheduled to arrive at this step
            # Using a while loop in case multiple spikes arrive at the same time
            while spike_queue and spike_queue[0][0] == step:
                _, target_idx, weight = spike_queue.popleft()
                # Add weight to the appropriate synaptic current
                if weight > 0: # Excitatory
                    self.reservoir_neurons[target_idx].i_syn_E += weight
                else: # Inhibitory
                    self.reservoir_neurons[target_idx].i_syn_I += weight
            
            # --- 3. Update All Reservoir Neurons ---
            for i in range(self.n_total):
                neuron = self.reservoir_neurons[i]
                did_spike = neuron.update()
                
                if did_spike:
                    # --- 4. Handle Fired Spikes ---
                    # a. Record the spike
                    self.spike_recorder.append((current_time_ms, i))
                    
                    # b. Propagate to recurrent connections (if any were defined)
                    # This part is left empty as the original code also removed recurrent connections
                    # for target_idx, weight, delay_steps in self.recurrent_connections[i]:
                    #     arrival_step = step + delay_steps
                    #     spike_queue.append((arrival_step, target_idx, weight))
            
            # --- 5. Record Data ---
            for i, neuron_idx in enumerate(neurons_to_record_v):
                self.voltage_recorder[i].append(self.reservoir_neurons[neuron_idx].v)
        
        print("Simulation finished.")

    def plot(self, duration_s, neurons_to_record_v):
        """
        Generates plots for the simulation results.
        """
        # --- Raster Plot ---
        if not self.spike_recorder:
            print("No spikes were recorded.")
            return
            
        spike_times, neuron_ids = zip(*self.spike_recorder)
        
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True,
                                       gridspec_kw={'height_ratios': [1, 2]})
        
        ax1.set_title('LSM Reservoir Activity (Raster Plot)')
        ax1.scatter(np.array(spike_times) / 1000.0, neuron_ids, marker='.', s=10, c='k', alpha=0.8)
        ax1.set_ylabel('Neuron Index')
        ax1.set_ylim(-1, self.n_total)
        # Add line to show input switch
        ax1.axvline(duration_s / 2, color='gray', linestyle='--', label='Input Switch')
        ax1.legend()

        # --- Voltage Traces ---
        ax2.set_title('Membrane Potential Traces (Subset)')
        times_s = np.arange(0, duration_s, self.dt / 1000.0)
        
        for i, neuron_idx in enumerate(neurons_to_record_v):
            # Ensure the recorded voltage length matches the time array length
            recorded_v = self.voltage_recorder[i]
            ax2.plot(times_s[:len(recorded_v)], recorded_v, label=f'Neuron {neuron_idx}')
            
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Membrane Potential (mV)')
        ax2.legend(loc='upper right', fontsize='small')
        
        plt.tight_layout()
        plt.show()


# --- Main Execution ---
if __name__ == '__main__':
    # --- Simulation Setup ---
    DT = 1.0  # Timestep in ms
    SIM_DURATION = 10.0 # Simulation duration in seconds (for one half)
    TOTAL_DURATION = SIM_DURATION * 2

    # --- Neuron and Synapse Model Parameters ---
    LIF_PARAMS = {
        'v_rest': -65.0,   # mV
        'v_reset': -65.0,  # mV
        'v_thresh': -50.0, # mV
        'tau_m': 20.0,     # ms
        'tau_refrac': 5.0, # ms
        'tau_syn_E': 5.0,  # ms
        'tau_syn_I': 10.0, # ms
        'i_offset': 0.2,   # nA (constant background current)
    }

    # --- Network Architecture ---
    N_INPUT = 50
    N_EXC = 320
    N_INH = 80
    
    # --- Input Signal Parameters ---
    RATE_HIGH = 100.0  # Hz
    RATE_LOW = 25.0    # Hz

    # --- Generate Input Spike Trains ---
    print("Generating input spike trains...")
    rng = np.random.default_rng(seed=42) # for reproducibility
    # High rate input for the first half of the simulation
    spikes_high = generate_poisson_spike_trains(N_INPUT, RATE_HIGH, 0.0, SIM_DURATION, dt=DT, rng=rng)
    # Low rate input for the second half
    spikes_low = generate_poisson_spike_trains(N_INPUT, RATE_LOW, SIM_DURATION, TOTAL_DURATION, dt=DT, rng=rng)
    print("Input generation complete.")

    # --- Create and Configure the Network ---
    lsm_network = Network(lif_params=LIF_PARAMS, n_input=N_INPUT, 
                          n_exc=N_EXC, n_inh=N_INH, dt=DT)
    
    # Define connections
    W_INPUT = 1.0  # Synaptic weight from input to reservoir (nA)
    DELAY_MS = 1.0 # Synaptic delay in ms
    lsm_network.connect_inputs(w_input=W_INPUT, delay_ms=DELAY_MS)
    
    # --- Run Simulation ---
    neurons_to_plot = list(range(10)) # Plot the first 10 neurons
    lsm_network.run(duration_s=TOTAL_DURATION,
                    input_spike_trains_high=spikes_high,
                    input_spike_trains_low=spikes_low,
                    neurons_to_record_v=neurons_to_plot)

    # --- Plot Results ---
    lsm_network.plot(duration_s=TOTAL_DURATION, neurons_to_record_v=neurons_to_plot)
