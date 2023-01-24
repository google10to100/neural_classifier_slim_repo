import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import tkinter
  

class Spikes:
    '''For a given set of raw time series, Spikes is a 3D array container for neural spikes at all electrodes'''

    def __init__(self, raw, stim_mode, plot_stim):
        # some constants
        self.mid_ei = 2
        self.spk_len = 200
        # self.stim_artifact_len = 23
        self.stim_artifact_len = 30
        self.noise_sf = 2.5
        self.artifact_sf = 1.5
        self.sr = 100000
        self.pass_band = [100, 10000]
        self.plot_num = 100
        self.sig_colors  = ['g', 'b', 'y', 'r', 'm']
        
        # assign passed parameters
        self.raw = raw
        self.stim_mode = stim_mode
        self.plot_stim = plot_stim
        # derived parameters
        self.stim_sig = self.raw[:,0]
        self.neural_sigs = self.raw[:,1:]          
        self.neural_sigs_f = self.filter_neural()
        self.stim_events, self.stim_diff = self.get_stim_events()
        self.upper_bound, self.lower_bound, self.to_keep_is = self.get_good_spike_indexes()
        self.good_mid_spks = self.to_good_spikes(self.mid_ei)        
        self.spks_len = self.get_average_spike_length()
        self.num_good_spks = self.good_mid_spks.shape[1]

    def normalize_amplitude_only(self, spks):
        # normalize spikes by their peak-to-peak amplitudes only        
        scale_factor = abs(np.ptp(spks, axis=0))    # normalize spikes by thei peak-to-peak amplitudes
        spks_norm = spks/scale_factor
        return spks_norm

    def normalize(self, spks, norm_len):
        # normalize spikes by peak-to-peak amplitude and in time
        # scale all spikes to a standard width        
        
        sig_width = self.spks_len
        new_width = norm_len
        dat_range = range(sig_width)
    
        x = range(sig_width)
        new_x = np.linspace(0, sig_width, new_width)  
        y = spks[dat_range, :]

        n_spks = spks.shape[1]
        new_ys = np.zeros([new_width, n_spks])
        #print(new_ys.shape)
        for i in range(n_spks):        
            new_ys[:, i] = np.interp(new_x, x, y[:, i])
        #scale_factor = abs(np.min(new_ys, axis=0))   # normalize spikes by the magnitude of the negative peak
        scale_factor = abs(np.ptp(new_ys, axis=0))    # normalize spikes by thei peak-to-peak amplitudes
        #scale_factor = 1
        spks_norm = new_ys/scale_factor
        return spks_norm    

    def get_average_spike_length(self):
        spk_ref = self.to_median(self.good_mid_spks)
        neg_peak, neg_peak_i, hump2, hump2_i = self.get_peaks(spk_ref)
        sig_after_hump2 = spk_ref[hump2_i:]
        zero_xings = np.where(sig_after_hump2 < 0.01)
        first_zero_xing = zero_xings[0][0]
        # sig_end_i = hump2_i + first_zero_xing
        sig_end_i = hump2_i + 10
        return sig_end_i

    def get_peaks(self, spk):
        neg_peak = np.amin(spk)
        neg_peak_i = np.where(spk==neg_peak)[0][0]        
        hump2 = np.amax(spk[neg_peak_i:])
        hump2_i = np.where(spk==hump2)[0][0]
        return neg_peak, neg_peak_i, hump2, hump2_i        

    def to_median(self, spks):        
        spk_median = np.median(spks, axis=1)
        return spk_median

    def to_good_spikes(self, ei):
        # upper_bound, lower_bound, to_keep_is = self.get_good_spike_indexes()
        # to_keep_is = self.to_keep_is
        test_spks = self.to_spikes(ei)
        good_spks = test_spks[:, self.to_keep_is]
        return good_spks

    def get_good_spike_indexes(self):
        '''compute upper and lower bounds where legitimate neural spikes are likely to lie'''
        test_spks = self.to_spikes(self.mid_ei)
        # upper-bound based on the minimum of the background noise
        noise_band = range(-50,-1)
        noise_mins = np.min(test_spks[noise_band,:], axis=0)  # minimum of the last 50 data points - i.e in the noise band
        upper_bound = self.noise_sf * np.median(noise_mins)
        # lower-bound based on the signal minimum
        # avg_spk_len = self.get_average_spike_length()
        # sig_band = range(self.stim_artifact_len, avg_spk_len)
        sig_band = range(self.stim_artifact_len, self.stim_artifact_len + self.spk_len-50)
        sig_mins = np.min(test_spks[sig_band,:], axis=0)
        lower_bound = self.artifact_sf * np.median(sig_mins)
        # get indexes of good spikes
        to_keep_test = (sig_mins < upper_bound) & (sig_mins > lower_bound)
        to_keep_is = to_keep_test.nonzero()[0]
        # return
        return upper_bound, lower_bound, to_keep_is  

    def to_spikes(self, ei):
        stim_events = self.stim_events
        spk_len = self.spk_len
        stim_artifact_len = self.stim_artifact_len
        # num_sigs = stim_events.shape[0] - 1 # don't extract signals after the final stim event as there may be less than "sig_len" data points
        num_sigs = stim_events.shape[0]
        spks = np.zeros([spk_len, num_sigs])
        for stim_i in range(num_sigs):
            start_i = stim_events[stim_i] + stim_artifact_len
            stop_i = start_i + spk_len
            spk = self.neural_sigs_f[start_i:stop_i, ei]
            spks[:, stim_i] = spk        
        return spks     

    def plot_stim_snapshot(self):
        plt.figure(self.plot_num)
        plt.plot(self.stim_sig[:20000], 'k')
        plt.title('Stimulation signals snapshot')
        self.plot_num += 1

    def plot_neural_snapshot(self, stim_i, eis, sigs_to_plot):
        #sigs = self.neural_sigs_f
        if sigs_to_plot.lower() == 'raw':
            sigs = self.neural_sigs
        elif sigs_to_plot.lower() == 'filtered':
            sigs = self.neural_sigs_f
        start_i = self.stim_events[stim_i]
        end_i = start_i + self.spk_len
        plt.figure(self.plot_num)
        for ei in eis:
            plt.plot(sigs[start_i:end_i, ei], self.sig_colors[ei])
        plt.title('Neural signals snapshot')
        self.plot_num += 1

    def filter_neural(self):
        '''Filter neural signals using a 2nd order Butterworth filter'''
        sos = signal.butter(2, self.pass_band, 'bandpass', fs=self.sr, output='sos')  # butterworth filter characteristics
        neural_sigs_f = signal.sosfilt(sos, self.neural_sigs, axis=0)
        return neural_sigs_f        

    def get_stim_events(self):
        ''' get stimulation signals from raw data '''
        # stim_thresh = 2.5
        stim_dat = self.stim_sig
        stim_clean = stim_dat.astype('int')
        stim_diff = stim_clean[1:]-stim_clean[0:-1]
        # body - identify stimulus event locations
        # stim_thresh = np.amax(stim_clean) - 0.5
        stim_thresh = 2.5
        if self.stim_mode.lower() == "rise":
            #print(stim_thresh)
            stim_events_tuple = np.where(stim_diff > stim_thresh)
        elif self.stim_mode.lower() == "fall":
            #print(-stim_thresh)
            stim_events_tuple = np.where(stim_diff < -stim_thresh)
        else:
            stim_events_tuple = np.where(abs(stim_diff) > stim_thresh)
        stim_events = stim_events_tuple[0]
        #print(stim_events)
        # outputs
        if self.plot_stim:
            plt.figure(self.plot_num)
            plt.plot(stim_diff, 'k.')
            plt.plot(stim_events, stim_diff[stim_events], 'r.')
            self.plot_num += 1
            # plt.show(block=False)        
        # return
        stim_events = stim_events[1:-1]
        return stim_events, stim_diff