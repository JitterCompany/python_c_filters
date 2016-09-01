import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from collections import defaultdict
from functools import reduce

import cfilt


def get_decimation_filter(decimation_factor, gpass_dB=0.0000001, gstop_dB=60,
                          low_cut_off_factor=None, ftype='ellip'):
    """
    Returns decimation filter for factor
    """

    high_cut_off = 1/decimation_factor
    if low_cut_off_factor is None:
        low_cut_off_factor = 2
    low_cut_off = high_cut_off / low_cut_off_factor
    filt = signal.iirdesign(low_cut_off, high_cut_off, gpass_dB, 
                            gstop_dB, analog=False, ftype=ftype, output='sos')
    ba_filt = signal.iirdesign(low_cut_off, high_cut_off, gpass_dB,
                               gstop_dB, analog=False, ftype=ftype, output='ba')

    return filt, ba_filt

def get_cfilt32_response(filt_index, fs, freqs=None):
    """ 
    Calculate filter response using 32 bits filter (floats) from CFilt package.

    Parameters
    ----------
    filt_index: int
        filter handle for cfilt
    fs: int
        sample frequency
    freqs: arraylike, optional
        frequecies where the response is evaluated.
        default: range [1, 300]  

    Returns
    -------
    freqs: arraylike
        list of frequencies where response was evaluated
    h: arraylike, dtype: float32
        filter response 
    """

    if freqs is None:
        freqs = np.arange(300)
    T = 4
    t = np.arange(0, T, 1/fs)
    mag = []
    for f in freqs:
        x = np.sin(2*np.pi*f*t, dtype='float32')
        y = cfilt.filter32_apply(filt_index, x)
        mag.append(max(np.abs(y[2*fs:])))
    return freqs, np.array(mag)

def get_cfilt64_response(filt_index, fs, freqs=np.arange(300)):
    """ 
    Calculate filter response using 64 bits filter (doubles) from CFilt package.

    Parameters
    ----------
    filt_index: int
        filter handle for cfilt
    fs: int
        sample frequency
    freqs: arraylike, optional
        frequecies where the response is evaluated.
        default: range [1, 300]  

    Returns
    -------
    freqs: arraylike
        list of frequencies where response was evaluated
    h: arraylike, dtype: float64
        filter response 
    """
    T = 4
    t = np.arange(0, T, 1/fs)
    mag = []
    for f in freqs:
        x = np.sin(2*np.pi*f*t, dtype='float64')
        y = cfilt.filter64_apply(filt_index, x)
            
        mag.append(max(np.abs(y[2*fs:])))
    return freqs, np.array(mag)

def calc_better_response(filter_index, response_func, fs):
    freq_start = [0.01, 0.014, 0.0097, 0.089, 0.013]

    hs = []
    for start in freq_start:
        freqs = np.arange(start, fs/2, 0.09)
        _, h = response_func(filter_index, fs, freqs)
        hs.append(h)
    
    l = min([len(h) for h in hs])
    hs = [h[:l] for h in hs]
    h = reduce(np.maximum, hs[1:], hs[0])
    freqs = np.linspace(0.01, fs/2, l)
    return freqs, h

def plot_filter_response(freqs, mag, fig=None, label=None, fs=None):

    if not fig:
        plt.figure(figsize=(16,10))
        
    db = 20*np.log10(np.abs(mag))
    plt.plot(freqs, db, label=label)
    plt.grid(True)
    plt.ylabel('Amplitude [dB]', color='b')
    plt.xlabel('Frequency ' + ('[Hz]'if fs else '[rad/sample]' ))
    
def optimal_filter(q, fs):
    results = defaultdict(list)
    for n in range(2,62):
        f, ba = get_decimation_filter(q, gpass_dB=0.00001, gstop_dB=60, 
                                      low_cut_off_factor=n, ftype='ellip')
        a_f = np.abs(f)
        m = np.min(a_f[a_f>0])
        num_stages = len(f)
        results[num_stages].append((n, m, ba))

    ax = plt.figure().gca()
    ax.set_title('Trade-off curves for q = %d' % q)
    
    print("Optimal results for q = %d" % q)
    for k in results:
        x,y,data = zip(*results[k])
        i = np.argmin(y)
        print('Best choice with %d stages: low_cut_off_factor = %d, smallest value = %.6f' 
              %(k, x[i], y[i]))
        
        ax.plot(x,y, 'x-', label="%d stages" % k)

        fig2 = plt.figure()
        for d in data:
            w,h = signal.freqz(*d)
            plot_filter_response(w / np.pi * (fs/2), h, fig=fig2, fs=fs)
            plt.title('responses for %d stages at q = %d ' % (k,q))
    ax.legend()
    ax.grid(True)
