
import numpy as np
from scipy.ndimage import gaussian_filter1d



def make_psth(spike_times, stim_times, pre_window=0.5, post_window=1.0, bin_size=0.05):

    # Convert inputs to numpy arrays
    spike_times = np.array(spike_times)
    stim_times = np.array(stim_times)
    
    # Create time bins
    bins = np.arange(-pre_window, post_window + bin_size, bin_size)
    bin_centers = bins[:-1] + bin_size/2
    
    # Initialize array to store spike counts for each trial
    all_counts = np.zeros((len(stim_times), len(bins)-1))
    
    # Count spikes in each bin for each trial
    for i, stim_time in enumerate(stim_times):
        # Find spikes within the window for this trial
        mask = ((spike_times >= stim_time - pre_window) & 
                (spike_times < stim_time + post_window))
        trial_spikes = spike_times[mask] - stim_time
        
        # Count spikes in each bin
        counts, _ = np.histogram(trial_spikes, bins=bins)
        all_counts[i, :] = counts
    
    # Convert counts to firing rate (spikes/second)
    firing_rates = all_counts / bin_size
    
    return firing_rates, bin_centers



def smooth_psth(psth, bin_size, kernel_type='gaussian', sigma=None, window_size=None):
    if sigma is None:
        sigma = 3 * bin_size  # Default: 3 bins
    if window_size is None:
        window_size = 6 * sigma  # Default: 6 sigma
        
    # Convert time units to bins
    sigma_bins = int(sigma / bin_size)
    window_bins = int(window_size / bin_size)
    
    # Ensure window size is odd
    if window_bins % 2 == 0:
        window_bins += 1
    
    if kernel_type.lower() == 'gaussian':
        # Use scipy's optimized gaussian filter
        smoothed = gaussian_filter1d(psth, sigma_bins)
    elif kernel_type.lower() == 'boxcar':
        kernel = boxcar_kernel(window_bins)
        smoothed = np.convolve(psth, kernel, mode='same')
    else:
        raise ValueError("kernel_type must be 'gaussian' or 'boxcar'")
        
    return smoothed