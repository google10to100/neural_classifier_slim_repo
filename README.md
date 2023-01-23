# neural_slassifier_slim_repo

# OVERVIEW
Classifies the directionality of recorded neural signals - i.e. "spikes" - using a neural network

# OPERATING INSTRUCIONS
1. Run "make_spike_training_data.ipynb"
	a. Toggle the "ei" parameter in the "Controls" section to change the electrode number used to record signals.
		- Note: possible values are [0, 1, 2, 3, 4] - to used to record the signals
		- Note: ei=2 means left- and right-bound signals were recorded at the mid-channel location, where differences between left- and right-bound spikes are expected to be small - i.e. spike shapes are expected to be similar and difficult to classify
		- Note: ei=0 or 1 means left-bound spikes will be recorded as having a sharper negative peak and are easier to distinguish from their right-bound counterparts
		- Note: ei=3 or 4 means right-bound spikes will be recorded as having a sharper negative peak and are easier to distinguish from their left-bound counterparts
	b. Output of this program are the variables "X" and "y", where X is a 2D array of spikes and "y" their true direction labels.
		- Note: in "y", 0=left-bound, 1=right-bound
		- Note: in "X", each row represents a different spike; columns hold the actual time-varying signal 

2. Run "nn_for_spike_data_notebook.ipynb"
	a. "scale_dat" can be toggled to 0 or 1. If "1", spike amplitudes are divided by 100 fold. This puts stress of the neural network model and makes it more difficult to classify spikes correctly.  If "0", spikes are left to the amplitudes in the "X.npy" array, output step (1) above
	b. Output of this program is a confusion matrix (variable "cm") and a print statement at the end which states:
		"Result = [aaa] misclassified spikes out of [AAA] total.", where "aaa" and "AAA" are the number of misclassified spikes and total number of spikes, respectively.

