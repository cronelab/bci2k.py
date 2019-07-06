# Import modules
import os
import sys
import copy
import json
from collections import OrderedDict
import warnings
import functools
import inspect
import numpy as np
np.mode = lambda x, axis = None: np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis, x)
eps = 1e-9
log_eps = np.log(np.finfo(np.float64).eps)
import scipy
import scipy.io
import scipy.io.wavfile
import sympy
from fractions import Fraction
import h5py
import mne
import matplotlib
from matplotlib import pyplot as plt
from ecogpy.python.pbar import pbar
from ecogpy.python.h5eeg import H5EEGFile
import xarray as xr
import xcog
import xcog.pipeline
import xcog.plot as xplot
import xarray as xr


# Densenet
import numpy as np 
from mne.filter import filter_data
import pyedflib
import scipy.io.wavfile as wav
import scipy
import scipy.signal
import MelFilterBank as mel
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.stats import mode
import os
import librosa
import lws
import librosa.filters
import math


# Define lower-level functions
# Correct numpy rounding plus type conversion
def np_round(x):
    if not type(x) in (list, tuple, np.ndarray):
        scalar = True
        x = np.array([x])
    else:
        scalar = False
    rup = np.where(x - np.floor(x) >= 0.5)
    rdown = np.where(x - np.floor(x) < 0.5)
    if rup:
        x[rup] = np.ceil(x[rup])
    if rdown:
        x[rdown] = np.floor(x[rdown])
    x = x.astype(int)
    if scalar:
        x = x[0]
    return x
# Extensible log plus c
def logpc(x, c = 0):
    return np.log(x + c)
# Extensible log minus c
def expmc(x, c = 0):
    return np.exp(x - c)



def preprocess(raw_data, channel_count, element_count, fs =1000, eeg_fs = 1000, extract_mode = 'others', rmline = True, use_fft = False, fft_size = 50, hg = slice(70, 170), lf = slice(10, 50), pad = True, doub_pad = True, post_pad = True, log_c = 0, diff = True, diff_order = 2, norm_mode = 'running', norm_blockwise = False, norm_win = 10, apply_CAR = False, detrend = False):
    # change input data into an xarray with time and channel dimension
    times = list(range(element_count))
    channels = list(range(channel_count))
    raw_data = xr.DataArray(raw_data, dims = ['time','channel'], coords = {'time': times, 'channel': channels, 'fs':1000})
    
    #Spectral parameters
    fft_shift = fft_size/5 # ms
    fft_size = int(fft_size/1e3*eeg_fs)
    fft_shift = int(fft_shift/1e3*eeg_fs)

    spectrogram_params = {
        'fs': int(fs),
        'nperseg': int(fft_size),
        'noverlap': int(fft_size - fft_shift)
    }

    diff_win = fft_size/2/eeg_fs/1e3 # ms
    diff_win = diff_win*1e3
    norm_win = int(norm_win*1000/fft_shift)


    # Adaptative CAR if needed
    if apply_CAR:
        raw_data = xcog.pipeline.adaptive_CAR(raw_data)

    # Detrend signal if needed
    if detrend:
        if block_trans_time:
            bp = block_trans_time
        else:
            bp = 0
        raw_data = xcog.pipeline.detrend(raw_data, bp = bp)

    # Filter line noise including harmonics if needed
    if rmline:
        raw_data = xcog.pipeline.remove_linenoise(raw_data)


    # Extract raw data based on extraction mode
    if extract_mode == 'trial' and (norm_mode == 'global' or norm_mode == 'local' or norm_mode == ''):
        raw_data_ext = xcog.pipeline.extract_events(raw_data, trial_events, win = trl, fs = eeg_fs)
        trange = trl
    elif extract_mode == 'word' and (norm_mode == 'global' or norm_mode == 'local' or norm_mode == ''):
        raw_data_ext = xcog.pipeline.extract_events(raw_data, word_trial_events, win = wrd_trl, fs = eeg_fs)
        trange = wrd_trl
    elif extract_mode == 'phone' and (norm_mode == 'global' or norm_mode == 'local' or norm_mode == ''):
        raw_data_ext = xcog.pipeline.extract_events(raw_data, phone_trial_events, win = phn_trl, fs = eeg_fs)
        trange = phn_trl
    else:
        raw_data_ext = raw_data
        trange = None

    # Get trial correspondences of block transitions
    if extract_mode == 'trial':
        block_trans = [[]]*len(block_trans_time)
        for i in range(len(block_trans_time)):
            block_trans[i] = np.where(np.array([t/eeg_fs for _, t, _ in trial_events]) < block_trans_time[i])[0][-1] + 2
    elif extract_mode == 'word':
        block_trans = [[]]*len(block_trans_time)
        for i in range(len(block_trans_time)):
            block_trans[i] = np.where(np.array([t/eeg_fs for _, t, _ in word_trial_events]) < block_trans_time[i])[0][-1] + 2
    elif extract_mode == 'phone':
        block_trans = [[]]*len(block_trans_time)
        for i in range(len(block_trans_time)):
            block_trans[i] = np.where(np.array([t/eeg_fs for _, t, _ in phone_trial_events]) < block_trans_time[i])[0][-1] + 2

    # Extract spectrogram
    if use_fft:
        tf_data = xcog.pipeline.timefreq_fft(raw_data, pad = pad, doub_pad = doub_pad, post_pad = post_pad, **spectrogram_params)
        tf_data = logpc(tf_data, c = log_c)
    else:
        if doub_pad:
            tpad = (fft_size - fft_shift)/eeg_fs
        else:
            tpad = 0
        tf_data = xcog.pipeline.bandpass_filterbank(raw_data, nbands = int(fft_size//2 + 1), frange = [0, int(eeg_fs/2)],
                                                    tpad = tpad)
        tf_data = xcog.pipeline.rolling_transform(lambda x: logpc( np.mean(np.square(x), axis = 0), c = 0), tf_data,
                                                fft_size, stride = fft_shift, pad = True)
    tf_data.values[np.isnan(tf_data.values)] = log_eps
    tf_data.values[np.isinf(tf_data.values)] = log_eps

    if norm_blockwise and block_trans_time:
        tf_tmp = [[]]*(len(block_trans) + 1)
        if norm_mode == 'global':
            for i in range(len(block_trans) + 1):
                if not i:
                    tf_tmp[i] = tf_data.sel(trial = slice(1, block_trans[i] - 1))
                elif i == len(block_trans):
                    tf_tmp[i] = tf_data.sel(trial = slice(block_trans[i - 1], tf_data.trial.values[-1]))
                else:
                    tf_tmp[i] = tf_data.sel(trial = slice(block_trans[i - 1], block_trans[i] - 1))
        elif norm_mode == 'overall':
            for i in range(len(block_trans) + 1):
                if not i:
                    tf_tmp[i] = tf_data.sel(time = slice(tf_data.time.values[0], block_trans_time[i] - 1/tf_data.fs.values/2))
                elif i == len(block_trans):
                    tf_tmp[i] = tf_data.sel(time = slice(block_trans_time[i - 1], tf_data.time.values[-1]))
                else:
                    tf_tmp[i] = tf_data.sel(time = slice(block_trans_time[i - 1], block_trans_time[i] - 1/tf_data.fs.values/2))
    else:
        tf_tmp = [tf_data]
    if norm_mode == 'global':
        for i in range(len(tf_tmp)):
            tf_tmp[i] = xcog.pipeline.normalize(tf_tmp[i], tf_tmp[i].sel(time = slice(*(trange[0], 0 - 1/tf_data.fs.values/2))))
    elif norm_mode == 'local':
        for i in range(len(tf_data.trial)):
            tf_data[..., i] = xcog.pipeline.normalize(tf_data[..., i],
                                                    tf_data[..., i].sel(time = slice(*(trange[0], 0 - 1/tf_data.fs.values/2))))
    elif norm_mode == 'overall':
        for i in range(len(tf_tmp)):
            tf_tmp[i] = xcog.pipeline.normalize(tf_tmp[i], tf_tmp[i])
    elif norm_mode == 'running':
        tf_data = xcog.pipeline.normalize(tf_data, norm_win, running = True)
    if norm_mode == 'global':
        tf_data = xr.concat(tf_tmp, dim = 'trial')
    elif norm_mode == 'overall':
        tf_data = xr.concat(tf_tmp, dim = 'time')

    # Extract high gamma
    if use_fft:
        hg_data = tf_data.sel(frequency = hg).mean('frequency')
    else:
        if doub_pad:
            tpad = (fft_size - fft_shift)/eeg_fs
        else:
            tpad = 0
        hg_data = xcog.pipeline.bandpass_filterbank(raw_data, bands = [hg.start, hg.stop], tpad = tpad)
        hg_data = xcog.pipeline.rolling_transform(lambda x: logpc( np.mean(np.square(x), axis = 0), c = 0), hg_data,
                                                fft_size, stride = fft_shift, pad = True)
    if norm_blockwise and block_trans_time:
        hg_tmp = [[]]*(len(block_trans) + 1)
        if norm_mode == 'global':
            for i in range(len(block_trans) + 1):
                if not i:
                    hg_tmp[i] = hg_data.sel(trial = slice(1, block_trans[i] - 1))
                elif i == len(block_trans):
                    hg_tmp[i] = hg_data.sel(trial = slice(block_trans[i - 1], hg_data.trial.values[-1]))
                else:
                    hg_tmp[i] = hg_data.sel(trial = slice(block_trans[i - 1], block_trans[i] - 1))
        elif norm_mode == 'overall':
            for i in range(len(block_trans) + 1):
                if not i:
                    hg_tmp[i] = hg_data.sel(time = slice(hg_data.time.values[0], block_trans_time[i] - 1/hg_data.fs.values/2))
                elif i == len(block_trans):
                    hg_tmp[i] = hg_data.sel(time = slice(block_trans_time[i - 1], hg_data.time.values[-1]))
                else:
                    hg_tmp[i] = hg_data.sel(time = slice(block_trans_time[i - 1], block_trans_time[i] -  block_trans_time[i] - 1/tf_data.fs.values/2))
    else:
        hg_tmp = [hg_data]
    if norm_mode == 'global' and len(hg_tmp) > 1:
        for i in range(len(hg_tmp)):
            hg_tmp[i] = xcog.pipeline.normalize(hg_tmp[i], hg_tmp[i].sel(time = slice(*(trange[0], 0 - 1/tf_data.fs.values/2))))
    elif norm_mode == 'local' and len(hg_tmp) > 1:
        for i in range(len(hg_data.trial)):
            hg_data[..., i] = xcog.pipeline.normalize(hg_data[..., i],
                                                    hg_data[..., i].sel(time = slice(*(trange[0], 0 - 1/tf_data.fs.values/2))))
    elif norm_mode == 'overall':
        for i in range(len(hg_tmp)):
            hg_tmp[i] = xcog.pipeline.normalize(hg_tmp[i], hg_tmp[i])
    elif norm_mode == 'running':
        hg_data = xcog.pipeline.normalize(hg_data, norm_win, running = True)
    if norm_mode == 'global':
        hg_data = xr.concat(hg_tmp, dim = 'trial')
    elif norm_mode == 'overall':
        hg_data = xr.concat(hg_tmp, dim = 'time')

    # Extract low frequency
    if use_fft:
        lf_data = tf_data.sel(frequency = lf).mean('frequency')
    else:
        if doub_pad:
            tpad = (fft_size - fft_shift)/eeg_fs
        else:
            tpad = 0
        lf_data = xcog.pipeline.bandpass_filterbank(raw_data, bands = [lf.start, lf.stop], tpad = tpad)
        lf_data = xcog.pipeline.rolling_transform(lambda x: logpc( np.mean(np.square(x), axis = 0), c = 0), lf_data,
                                                fft_size, stride = fft_shift, pad = True)
    if norm_blockwise and block_trans_time:
        lf_tmp = [[]]*(len(block_trans) + 1)
        if norm_mode == 'global':
            for i in range(len(block_trans) + 1):
                if not i:
                    lf_tmp[i] = lf_data.sel(trial = slice(1, block_trans[i] - 1))
                elif i == len(block_trans):
                    lf_tmp[i] = lf_data.sel(trial = slice(block_trans[i - 1], lf_data.trial.values[-1]))
                else:
                    lf_tmp[i] = lf_data.sel(trial = slice(block_trans[i - 1], block_trans[i] - 1))
        elif norm_mode == 'overall':
            for i in range(len(block_trans) + 1):
                if not i:
                    lf_tmp[i] = lf_data.sel(time = slice(lf_data.time.values[0], block_trans_time[i] - 1/lf_data.fs.values/2))
                elif i == len(block_trans):
                    lf_tmp[i] = lf_data.sel(time = slice(block_trans_time[i - 1], lf_data.time.values[-1]))
                else:
                    lf_tmp[i] = lf_data.sel(time = slice(block_trans_time[i - 1], block_trans_time[i] -  block_trans_time[i] - 1/tf_data.fs.values/2))
    else:
        lf_tmp = [lf_data]
    if norm_mode == 'global' and len(lf_tmp) > 1:
        for i in range(len(lf_tmp)):
            lf_tmp[i] = xcog.pipeline.normalize(lf_tmp[i], lf_tmp[i].sel(time = slice(*(trange[0], 0 - 1/tf_data.fs.values/2))))
    elif norm_mode == 'local' and len(lf_tmp) > 1:
        for i in range(len(lf_data.trial)):
            lf_data[..., i] = xcog.pipeline.normalize(lf_data[..., i],
                                                    lf_data[..., i].sel(time = slice(*(trange[0], 0 - 1/tf_data.fs.values/2))))
    elif norm_mode == 'overall':
        for i in range(len(lf_tmp)):
            lf_tmp[i] = xcog.pipeline.normalize(lf_tmp[i], lf_tmp[i])
    elif norm_mode == 'running':
        lf_data = xcog.pipeline.normalize(lf_data, norm_win, running = True)
    if norm_mode == 'global':
        lf_data = xr.concat(lf_tmp, dim = 'trial')
    elif norm_mode == 'overall':
        lf_data = xr.concat(lf_tmp, dim = 'time')
    return tf_data 














