import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import uproot
import awkward as ak
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import time

def read_root_files(filepath, cosmics=False):
    file = uproot.open(filepath)
    print(file.keys())
    analyszer = 'PNSCRPanalyzercosmics' if cosmics else 'PNSCRPanalyzerflashes1'
    evts = uproot.open(f"{filepath}:{analyszer}/Event")
    akw = evts.arrays(filter_name=evts.keys(), library="ak")
    print('number of trigger records:', len(akw['event']))
    return akw

def save_to_pkl(akw, output_filename):
    with open(output_filename, "wb") as f:
        pickle.dump(akw, f)

def get_info(akw, display=False):
    # akw['waveformsADC'][trigRec][chID]
    Nevents    = len(akw['waveformsADC'])
    TrigWindow = len(akw['waveformsADC'][0][0])
    Nchannels  = len(akw['waveformsADC'][0])
    maxima = []
    for chID in range(0, Nchannels):
        maxima.append(ak.max( akw['waveformsADC'][:][chID]) )
    if display:
        print(' events', Nevents)
        print(' trigger window', TrigWindow)
        print(' number of channels', Nchannels)
        print(' max height', int(np.max(maxima)) )
    return Nevents, TrigWindow, Nchannels, maxima

def subtract_pedestals_all_channels_np(akw, nmax=None):
    waveforms = akw['waveformsADC']
    nevents = len(waveforms) if nmax is None else min(nmax, len(waveforms))
    waveforms = waveforms[:nevents]

    # Determine number of channels and maximum waveform length
    n_channels = max(len(evt) for evt in waveforms)
    max_len = max(len(ch) for evt in waveforms for ch in evt)

    # Preallocate output arrays
    waveformsADC_np = np.full((nevents, n_channels, max_len), np.nan, dtype=np.float32)
    waveformsSubtracted_np = np.full((nevents, n_channels, max_len), np.nan, dtype=np.float32)
    pedestals_np = np.zeros((nevents, n_channels), dtype=np.int32)

    for i_evt, evt in enumerate(tqdm(waveforms, desc="Subtracting pedestals")):
        for i_ch, ch in enumerate(evt):
            ch_np = np.asarray(ch, dtype=np.int32)
            hist = np.bincount(ch_np)
            pedestal = np.argmax(hist)
            pedestals_np[i_evt, i_ch] = pedestal
            sub = ch_np - pedestal
            waveformsADC_np[i_evt, i_ch, :len(ch_np)] = ch_np
            waveformsSubtracted_np[i_evt, i_ch, :len(sub)] = sub

    # Wrap into awkward array with fields
    result = ak.Array({
        'waveformsADC': waveformsADC_np,
        'waveformsSubtracted': waveformsSubtracted_np,
        'pedestals': pedestals_np,
    })
    return result

if __name__=="__main__":
    cernbox = '/Users/emiliebertholet/cernbox/coldbox_data'

    ## PNS
    filepath = f'{cernbox}/anaCRP_files/ana_pns_small.root' 
    akw_pns = read_root_files(filepath, cosmics=False)
    print('\nPNS')
    get_info(akw_pns, display=True)
    result = subtract_pedestals_all_channels_np(akw_pns, nmax=None)
    save_to_pkl(result, 'akw_pns_pedSub.pkl')

    ## Cosmics
    filepath = f'{cernbox}/anaCRP_files/ana_cosmic_small.root' 
    akw_cos = read_root_files(filepath, cosmics=True)
    print('\nCosmics')
    get_info(akw_cos, display=True)
    result = subtract_pedestals_all_channels_np(akw_cos, nmax=None)
    save_to_pkl(result, 'akw_cos_pedSub.pkl')




