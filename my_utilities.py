import glob
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def load_raw(dat_dir, fnum, do_plot):
    '''loads specific .mat file from desired directory'''

    # constants
    f_ext = '*.mat'

    # body
    fnames = glob.glob(dat_dir + f_ext)
    fname_w_path = fnames[fnum]
    fname = fname_w_path.replace(dat_dir, "")
    f_contents = loadmat(fname_w_path)
    raw = f_contents['raw']

    # outputs
    print(f'\nLoading file {fnum} ... "{fname}" with size {raw.shape}\n')

    if do_plot:
        plt.figure(1)
        plt.plot(raw[1:5000,:])
        plt.title(fname)
        # plt.show(block=False)

    # return
    return raw, fname


def get_stim_events(raw, stim_mode, do_plot):
    ''' get stimulation signals from raw data '''

    stim_sig = raw[:, 0]
    stim_clean = stim_sig.astype('int')
    stim_diff = stim_clean[1:]-stim_clean[0:-1]

    # body
    # identify stimulus event locations
    stim_thresh = np.amax(stim_clean) - 0.5
    if stim_mode.lower() == "rise":
        stim_events_tuple = np.where(stim_diff > stim_thresh)
    elif stim_mode.lower() == "fall":
        stim_events_tuple = np.where(stim_diff < -stim_thresh)
    else:
        stim_events_tuple = np.where(abs(stim_diff) > stim_thresh)
    stim_events = stim_events_tuple[0]

    # outputs
    if do_plot:
        plt.figure(2)
        plt.plot(stim_diff, 'k.')
        plt.plot(stim_events, stim_diff[stim_events], 'r.')
        # plt.show(block=False)

    # return
    return stim_events, stim_diff


def get_label(fname):
    # returns 0 if neural signal is right-bound and 1 if left-bound
    if 'rbound' in fname.lower():
        label = 1
    elif 'lbound' in fname.lower():
        label = 0
    else:
        print('Error - "left" or "right" not found in file name.')
    return label

def normalize(spks):
    # normalize spikes by peak-to-peak amplitude and in time
    # scale all spikes to a standard width    
    sig_width = 85
    new_width = 100
    dat_range = range(sig_width)    
    x = range(sig_width)
    new_x = np.linspace(0, sig_width, new_width)
    y = spks[dat_range, :]
    n_spks = spks.shape[1]
    spks_norm = np.zeros([new_width, n_spks])
    for i in range(n_spks):
        new_y = np.interp(new_x, x, y[:, i])        
        sig_height = abs(np.min(y))
        spks_norm[:, i] = new_y/sig_height  # normalize to the magnitude of the negative peak
    return spks_norm