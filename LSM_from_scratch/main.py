from lsm_core import Network
from data_utils import generate_mock_spike_counts, convert_spike_counts_to_lsm_input
from analysis import plot_activity_and_pca
import numpy as np

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
    lsm_network.run(duration_s=TOTAL_DURATION, input_spike_trains=lsm_input_spikes)
    # --- 4. Plot Results ---
    plot_activity_and_pca(lsm_network, TOTAL_DURATION, lsm_input_spikes, hidden_indices, output_indices) 