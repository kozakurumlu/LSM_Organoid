import numpy as np
from collections import deque

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


class Network:
    """Manages the entire simulation, including all neurons and connections."""
    def __init__(self, lif_params, n_input, n_exc, n_inh, dt=1.0, rng=None):
        self.dt = dt
        self.n_input = n_input
        self.n_exc = n_exc
        self.n_inh = n_inh
        self.n_total = n_exc + n_inh
        self.reservoir_neurons = [LIFNeuron(lif_params, dt) for _ in range(self.n_total)]
        self.rng = rng if rng is not None else np.random.default_rng()
        
        self.connections = {
            'input': [[] for _ in range(n_input)],
            'recurrent': [[] for _ in range(self.n_total)]
        }

    def connect_inputs(self, w_input, delay_ms, p_connect=0.1):
        """Connects input neurons to the reservoir."""
        print(f"Connecting inputs with {p_connect*100}% probability...")
        delay_steps = int(delay_ms / self.dt)
        for i in range(self.n_input):
            for j in range(self.n_total):
                if self.rng.random() < p_connect:
                    self.connections['input'][i].append((j, w_input, delay_steps))

    def connect_reservoir(self, weights, delay_ms, p_connect):
        """Creates sparse, random connections within the reservoir (E-E, E-I, I-E, I-I)."""
        print("Connecting reservoir neurons internally with E/I balance...")
        delay_steps = int(delay_ms / self.dt)
        exc_neurons = range(self.n_exc)
        inh_neurons = range(self.n_exc, self.n_total)

        for i in range(self.n_total):
            is_pre_exc = (i < self.n_exc)
            for j in range(self.n_total):
                if i == j: continue
                
                is_post_exc = (j < self.n_exc)
                
                if self.rng.random() < p_connect:
                    if is_pre_exc and is_post_exc: weight = weights['ee']   # E -> E
                    elif is_pre_exc and not is_post_exc: weight = weights['ei'] # E -> I
                    elif not is_pre_exc and is_post_exc: weight = -weights['ie']# I -> E (negative)
                    else: weight = -weights['ii']                           # I -> I (negative)
                    
                    self.connections['recurrent'][i].append((j, weight, delay_steps))

    def run(self, duration_s, input_spike_trains):
        """Executes the simulation, now handling both E and I currents."""
        duration_ms = duration_s * 1000
        num_steps = int(duration_ms / self.dt)
        spike_queue = deque()
        self.spike_recorder = []
        
        print(f"Starting simulation trial for {duration_s}s...")
        
        for step in range(num_steps):
            current_time_ms = step * self.dt
            
            for neuron_idx, spike_times in enumerate(input_spike_trains):
                if np.any(np.isclose(spike_times, current_time_ms)):
                    for target_idx, weight, delay_steps in self.connections['input'][neuron_idx]:
                        spike_queue.append((step + delay_steps, target_idx, weight))

            while spike_queue and spike_queue[0][0] == step:
                _, target_idx, weight = spike_queue.popleft()
                if weight > 0:
                    self.reservoir_neurons[target_idx].i_syn_E += weight
                else: # Handle inhibitory currents
                    self.reservoir_neurons[target_idx].i_syn_I += weight
            
            for i in range(self.n_total):
                if self.reservoir_neurons[i].update():
                    self.spike_recorder.append((current_time_ms, i))
                    for target_idx, weight, delay_steps in self.connections['recurrent'][i]:
                        spike_queue.append((step + delay_steps, target_idx, weight))