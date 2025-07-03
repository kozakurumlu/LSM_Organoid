import pyNN.brian2 as pynn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from quantities import ms, s, Hz

# --- Optional: Suppress Brian2's compiler warnings if you haven't installed a C++ compiler ---
from brian2 import prefs
prefs.codegen.target = "numpy"

# --- Import TimedArray for time-varying input ---
from brian2 import TimedArray, defaultclock
from brian2 import second

# --- 1. Simulation Setup ---
pynn.setup(timestep=1.0)  # 1 ms for all groups

# --- 2. Neuron and Synapse Model Parameters ---
lif_params = {
    'v_rest': -65.0,
    'v_reset': -65.0,
    'v_thresh': -50.0,
    'tau_m': 20.0,
    'tau_refrac': 5.0,
    'tau_syn_E': 5.0,
    'tau_syn_I': 10.0,
    'i_offset': 0.2,  # added DC offset for excitability
}

# --- Helper function to generate Poisson spike times ---
def generate_poisson_spike_trains(n_neurons, rate, t_start, t_stop, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    spike_trains = []
    for _ in range(n_neurons):
        t = t_start
        spikes = []
        while t < t_stop:
            isi = rng.exponential(1.0 / rate)
            t += isi
            if t < t_stop:
                # Round to nearest ms, skip if duplicate
                t_ms = round(t * 1000.0)
                if len(spikes) == 0 or t_ms > spikes[-1]:
                    spikes.append(t_ms)
        spike_trains.append(np.array(spikes, dtype=float))  # ensure float dtype
    return spike_trains

# --- 3. Network Architecture ---
n_input = 50
rate_high = 100.0  # Hz (float, not Quantity)
rate_low = 25.0    # Hz

# Simulation duration in seconds (reduced for testing)
sim_duration = 10.0  # seconds
duration = sim_duration * s  # duration as a Quantity, for compatibility

# Generate spike trains for each input group (minimum interval 1 ms)
spike_trains_high = generate_poisson_spike_trains(n_input, rate_high, 0.0, sim_duration)
spike_trains_low = generate_poisson_spike_trains(n_input, rate_low, sim_duration, sim_duration * 2)

# Create SpikeSourceArray populations
input_high_rate = pynn.Population(
    n_input,
    pynn.SpikeSourceArray(spike_times=spike_trains_high),
    label="Input (100 Hz, first half)"
)
input_low_rate = pynn.Population(
    n_input,
    pynn.SpikeSourceArray(spike_times=spike_trains_low),
    label="Input (25 Hz, second half)"
)

n_exc = 320
n_inh = 80
n_total = n_exc + n_inh
# Use current-based LIF model for reservoir
reservoir = pynn.Population(
    n_total,
    pynn.IF_curr_alpha(**lif_params),
    label="Reservoir"
)
reservoir_exc = reservoir[:n_exc]
reservoir_inh = reservoir[n_exc:]

# Define subsets of reservoir neurons for each input
input_high_targets = reservoir[:n_input]  # First 50 neurons
input_low_targets = reservoir[n_input:2*n_input]  # Next 50 neurons

print(f"Input 100 Hz targets reservoir neurons: 0 to {n_input-1}")
print(f"Input 25 Hz targets reservoir neurons: {n_input} to {2*n_input-1}")

# --- 4. Synaptic Connections (Projections) ---
# Reduced weights to avoid over-excitation
w_input = 1.0  # nA for current-based model
w_ee = 0.01
w_ei = 0.0
w_ie = 0.0
w_ii = 0.0

input_connector = pynn.FixedProbabilityConnector(p_connect=1.0)  # increased connection probability
input_synapse = pynn.StaticSynapse(weight=w_input, delay=1.0) # delay is a float in ms

# Connect input_high_rate to input_high_targets only
proj_high = pynn.Projection(input_high_rate, input_high_targets, input_connector,
                receptor_type='excitatory', synapse_type=input_synapse)
# Connect input_low_rate to input_low_targets only
proj_low = pynn.Projection(input_low_rate, input_low_targets, input_connector,
                receptor_type='excitatory', synapse_type=input_synapse)

# Print number of synapses created for debugging
print(f"Input high-rate synapses: {proj_high.size()}")
print(f"Input low-rate synapses: {proj_low.size()}")

# Remove all recurrent connections for debugging
# (Comment out or remove all pynn.Projection lines for reservoir-reservoir connections)

# --- 5. Setup Recording ---
reservoir.record(['spikes', 'v'])  # Also record membrane potential
input_high_rate.record('spikes')
input_low_rate.record('spikes')

# --- 6. Run the Simulation ---
simulation_time = duration * 2
simulation_time_ms = simulation_time.rescale(ms).item()
pynn.run(simulation_time_ms)

# --- 7. Retrieve Data and Plot Raster (Subset) ---
reservoir_spikes = reservoir.get_data('spikes')
reservoir_v = reservoir.get_data('v')
input_high_spikes = input_high_rate.get_data('spikes')
input_low_spikes = input_low_rate.get_data('spikes')

# Plot input spike rasters for debugging
fig_input, ax_input = plt.subplots(figsize=(12, 4))
ax_input.set_title('Input Populations Raster Plot')
ax_input.set_xlabel('Time (s)')
ax_input.set_ylabel('Input Neuron Index')
for i, spiketrain in enumerate(input_high_spikes.segments[0].spiketrains):
    t = np.array(spiketrain.rescale(s))
    ax_input.plot(t, np.full_like(t, i), 'b.', markersize=2)
for i, spiketrain in enumerate(input_low_spikes.segments[0].spiketrains):
    t = np.array(spiketrain.rescale(s))
    ax_input.plot(t, np.full_like(t, i + n_input), 'r.', markersize=2)
ax_input.set_xlim(0, (duration*2).rescale(s).item())
ax_input.set_ylim(0, n_input*2)
plt.show()

# Plot reservoir spike raster for debugging
subset_neurons = 20
fig_res_raster, ax_res_raster = plt.subplots(figsize=(12, 4))
ax_res_raster.set_title('Reservoir Spike Raster (Subset)')
ax_res_raster.set_xlabel('Time (s)')
ax_res_raster.set_ylabel('Reservoir Neuron Index')
for i, spiketrain in enumerate(reservoir_spikes.segments[0].spiketrains[:subset_neurons]):
    t = np.array(spiketrain.rescale(s))
    ax_res_raster.plot(t, np.full_like(t, i), 'k.', markersize=2)
ax_res_raster.set_xlim(0, (duration*2).rescale(s).item())
ax_res_raster.set_ylim(0, subset_neurons)
plt.show()

# Plot raster for a subset of neurons for clarity
subset_neurons = 20  # Show only first 20 neurons
fig_raster, ax_raster = plt.subplots(figsize=(12, 6))
ax_raster.set_title('LSM Reservoir Activity (Raster Plot, Subset)')
ax_raster.set_xlabel('Time (s)')
ax_raster.set_ylabel('Neuron Index (subset)')
ax_raster.axvline(duration.item(), color='gray', linestyle='--', label='Input Switch')
ax_raster.text((duration - 25.0*s).item(), subset_neurons * 1.02, '100 Hz Input', ha='center')
ax_raster.text((duration + 25.0*s).item(), subset_neurons * 1.02, '25 Hz Input', ha='center')

for i, spiketrain in enumerate(reservoir_spikes.segments[0].spiketrains[:subset_neurons]):
    t = np.array(spiketrain.rescale(s))
    ax_raster.plot(t, np.full_like(t, i), '.', markersize=2, label=f'Neuron {i}' if i < 5 else None)

ax_raster.set_xlim(0, (duration*2).rescale(s).item())
ax_raster.set_ylim(0, subset_neurons)
ax_raster.legend(loc='upper right', fontsize='small', ncol=2)
plt.show()

# --- Voltage Plot for a Subset of Neurons ---
print("\nPlotting membrane potentials for a subset of neurons...")
fig_v, ax_v = plt.subplots(figsize=(12, 8))
ax_v.set_title('Membrane Potential Traces (Subset)')
ax_v.set_xlabel('Time (s)')
ax_v.set_ylabel('Membrane potential (mV)')

analogsignal = reservoir_v.segments[0].analogsignals[0]
t = analogsignal.times.rescale(s)
for i in range(subset_neurons):
    v = analogsignal[:, i]
    ax_v.plot(t, v, label=f'Neuron {i}')
ax_v.legend(loc='upper right', fontsize='small', ncol=2)
plt.show()


# --- 9. End Simulation ---
pynn.end()
print("\nSimulation complete.")

