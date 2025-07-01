import pyNN.brian2 as pynn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from quantities import ms, s, Hz

# --- Optional: Suppress Brian2's compiler warnings if you haven't installed a C++ compiler ---
from brian2 import prefs
prefs.codegen.target = "numpy"

# --- 1. Simulation Setup ---
pynn.setup(timestep=1.0)

# --- 2. Neuron and Synapse Model Parameters ---
lif_params = {
    'v_rest': -65.0,
    'v_reset': -65.0,
    'v_thresh': -50.0,
    'tau_m': 20.0,
    'tau_refrac': 5.0,
    'tau_syn_E': 5.0,
    'tau_syn_I': 10.0,
}

# --- 3. Network Architecture ---
n_input = 50
rate_high = 100.0 * Hz
rate_low = 25.0 * Hz
duration = 100.0 * s  # Increased from 10s to 100s

input_high_rate = pynn.Population(
    n_input,
    pynn.SpikeSourcePoisson(rate=rate_high, start=0.0*ms, duration=duration),
    label="Input (100 Hz)"
)
input_low_rate = pynn.Population(
    n_input,
    pynn.SpikeSourcePoisson(rate=rate_low, start=duration, duration=duration),
    label="Input (25 Hz)"
)

n_exc = 320
n_inh = 80
n_total = n_exc + n_inh
reservoir = pynn.Population(
    n_total,
    pynn.IF_cond_alpha(**lif_params),
    label="Reservoir"
)
reservoir_exc = reservoir[:n_exc]
reservoir_inh = reservoir[n_exc:]

# Define subsets of reservoir neurons for each input
input_high_targets = reservoir[:n_input]  # First 50 neurons
input_low_targets = reservoir[n_input:2*n_input]  # Next 50 neurons

# --- 4. Synaptic Connections (Projections) ---
w_input = 0.05
w_ee = 0.03
w_ei = 0.02
w_ie = 0.015
w_ii = 0.025

input_connector = pynn.FixedProbabilityConnector(p_connect=0.1)
input_synapse = pynn.StaticSynapse(weight=w_input, delay=1.0) # delay is a float in ms

# Connect input_high_rate to input_high_targets only
pynn.Projection(input_high_rate, input_high_targets, input_connector,
                receptor_type='excitatory', synapse_type=input_synapse)
# Connect input_low_rate to input_low_targets only
pynn.Projection(input_low_rate, input_low_targets, input_connector,
                receptor_type='excitatory', synapse_type=input_synapse)

ee_synapse = pynn.StaticSynapse(weight=w_ee, delay=pynn.RandomDistribution('uniform', [1.0, 3.0]))
ei_synapse = pynn.StaticSynapse(weight=w_ei, delay=pynn.RandomDistribution('uniform', [1.0, 3.0]))
ie_synapse = pynn.StaticSynapse(weight=w_ie, delay=pynn.RandomDistribution('uniform', [1.0, 3.0]))
ii_synapse = pynn.StaticSynapse(weight=w_ii, delay=pynn.RandomDistribution('uniform', [1.0, 3.0]))

pynn.Projection(reservoir_exc, reservoir_exc, pynn.FixedProbabilityConnector(0.2), ee_synapse, receptor_type='excitatory')
pynn.Projection(reservoir_exc, reservoir_inh, pynn.FixedProbabilityConnector(0.3), ei_synapse, receptor_type='excitatory')
pynn.Projection(reservoir_inh, reservoir_exc, pynn.FixedProbabilityConnector(0.4), ie_synapse, receptor_type='inhibitory')
pynn.Projection(reservoir_inh, reservoir_inh, pynn.FixedProbabilityConnector(0.1), ii_synapse, receptor_type='inhibitory')

# --- 5. Setup Recording ---
reservoir.record(['spikes', 'v'])  # Also record membrane potential

# --- 6. Run the Simulation ---
simulation_time = duration * 2

# ==================== THIS IS THE FIX ====================
# Convert the simulation_time object (e.g., 20.0*s) to a float in ms
simulation_time_ms = simulation_time.rescale(ms).item()

# Now pass the simple float to the run() function
pynn.run(simulation_time_ms)
# =========================================================

# --- 7. Retrieve Data and Plot Raster ---
reservoir_spikes = reservoir.get_data('spikes')
reservoir_v = reservoir.get_data('v')

fig_raster, ax_raster = plt.subplots(figsize=(12, 6))
ax_raster.set_title('LSM Reservoir Activity')
ax_raster.set_xlabel('Time (s)')
ax_raster.set_ylabel('Neuron Index')
ax_raster.axvline(duration.item(), color='gray', linestyle='--')
ax_raster.text((duration - 25.0*s).item(), n_total * 1.02, '100 Hz Input', ha='center')
ax_raster.text((duration + 25.0*s).item(), n_total * 1.02, '25 Hz Input', ha='center')

for i, spiketrain in enumerate(reservoir_spikes.segments[0].spiketrains):
    t = np.array(spiketrain.rescale(s))
    ax_raster.plot(t, np.full_like(t, i), '.', markersize=2)

ax_raster.set_xlim(0, (duration*2).rescale(s).item())
ax_raster.set_ylim(0, n_total)
plt.show()

# --- Voltage Plot for a Subset of Neurons ---
print("\nPlotting membrane potentials for a subset of neurons...")
fig_v, ax_v = plt.subplots(figsize=(12, 8))
ax_v.set_title('Membrane Potential Traces (Subset)')
ax_v.set_xlabel('Time (s)')
ax_v.set_ylabel('Membrane potential (mV)')

# Plot for a subset (e.g., first 10 neurons)
subset_neurons = 10
for i in range(subset_neurons):
    v = reservoir_v.segments[0].filter(name='v')[0][i]
    t = reservoir_v.segments[0].analogsignals[0].times.rescale(s)
    ax_v.plot(t, v, label=f'Neuron {i}')
ax_v.legend()
plt.show()

# --- 8. PCA State-Space Trajectory Analysis ---
print("\nStarting PCA analysis...")
bin_width = 50.0 * ms
bins = np.arange(0, (duration*2).item(), bin_width.rescale(s).item()) # Ensure bin_width is also in seconds for histogram
binned_spikes = np.zeros((n_total, len(bins) - 1))

for i, spiketrain in enumerate(reservoir_spikes.segments[0].spiketrains):
    # Ensure spiketrain is also in seconds for histogram
    counts, _ = np.histogram(spiketrain.rescale(s), bins=bins)
    binned_spikes[i, :] = counts
binned_spikes = binned_spikes.T

pca = PCA(n_components=3)
trajectories = pca.fit_transform(binned_spikes)

print("Plotting 3D state-space trajectories...")
fig_pca = plt.figure(figsize=(10, 8))
ax_pca = fig_pca.add_subplot(111, projection='3d')
split_point = int((duration / bin_width).simplified.item())

ax_pca.plot(trajectories[:split_point, 0], trajectories[:split_point, 1], trajectories[:split_point, 2], label='100 Hz Input', color='crimson')
ax_pca.plot(trajectories[split_point:, 0], trajectories[split_point:, 1], trajectories[split_point:, 2], label='25 Hz Input', color='dodgerblue')
ax_pca.scatter(trajectories[0, 0], trajectories[0, 1], trajectories[0, 2], c='black', marker='o', s=50, label='Start')
ax_pca.scatter(trajectories[-1, 0], trajectories[-1, 1], trajectories[-1, 2], c='black', marker='X', s=50, label='End')

ax_pca.set_title('3D PCA of LSM State Trajectories')
ax_pca.set_xlabel('Principal Component 1')
ax_pca.set_ylabel('Principal Component 2')
ax_pca.set_zlabel('Principal Component 3')
ax_pca.legend()
plt.show()

# --- 9. End Simulation ---
pynn.end()
print("\nSimulation complete.")