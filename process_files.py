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
import h5py
import glob

import time

def read_root_files(filepath, cosmics=False):
    file = uproot.open(filepath)
    print(file.keys())
    analyszer = 'PNSCRPanalyzercosmics' if cosmics else 'PNSCRPanalyzerflashes1'
    evts = uproot.open(f"{filepath}:{analyszer}/Event")
    akw = evts.arrays(filter_name=evts.keys(), library="ak")
    # print('number of trigger records:', len(akw['event']))
    return akw

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

def subtract_pedestals_all_channels_np(akw, nmax=None, to_h5=False, filename='data'):
    waveforms = akw['waveformsADC']
    nevents   = len(waveforms) if nmax is None else min(nmax, len(waveforms))
    waveforms = waveforms[:nevents]

    # Determine number of channels and maximum waveform length
    n_channels = max(len(evt) for evt in waveforms)
    max_len    = max(len(ch) for evt in waveforms for ch in evt)

    # Preallocate output arrays
    waveformsADC_np = np.full((nevents, n_channels, max_len), np.nan, dtype=np.int32)
    pedestals_np = np.zeros((nevents, n_channels), dtype=np.int32)

    for i_evt, evt in enumerate(tqdm(waveforms, desc="Subtracting pedestals")):
        for i_ch, ch in enumerate(evt):
            ch_np = np.asarray(ch, dtype=np.int32)
            hist = np.bincount(ch_np)
            pedestal = np.argmax(hist)
            pedestals_np[i_evt, i_ch] = pedestal
            sub = ch_np - pedestal
            waveformsADC_np[i_evt, i_ch, :len(ch_np)] = ch_np

    if to_h5:
        with h5py.File(filename, "w") as f:
            f.create_dataset("waveformsADC", data=waveformsADC_np, compression="gzip")
            f.create_dataset("pedestals", data=pedestals_np)
    else:
        with open(filename, "wb") as f:
            pickle.dump({
                "waveformsADC": waveformsADC_np,
                "pedestals": pedestals_np,
            }, f)

def estimate_baseline_stability(waveform, max_deviation=5):
    median = np.median(waveform)
    mad = np.median(np.abs(waveform - median))

    quiet = np.abs(waveform - median) < max_deviation * mad

    if not np.any(quiet):  # if mask is empty
        return np.nan, np.nan, quiet ### maybe should return an arbitrary high number instead (e.g. 99999)

    baseline = np.mean(waveform[quiet])
    rms = np.std(waveform[quiet])
    return baseline, rms, quiet

def add_baseline_stability_estimate(sample):
    wfs = sample['waveformsADC']
    Nevents, _, Nchannels, _ = get_info(sample, display=False)
    sample['baseline_stability'] = np.array([[estimate_baseline_stability(wfs[trigID, chID], max_deviation=5)[1] for chID in range(Nchannels)] for trigID in range(Nevents)])


def main(sample, runNumber, savetoh5, input_files):
    for filepath in input_files:
        print('process file', filepath)
        info=filepath.split('/')[-1].split('.')[0].split('_')
        id_first, id_last = info[-2], info[-1]
        outputName = f'{sample}_nTuples_r{runNumber}_{id_first}-{id_last}'
        if sample=='pns':
            akw = read_root_files(filepath, cosmics=False)
            print('\nPNS')
        else:
            akw = read_root_files(filepath, cosmics=True)
            print('\nCosmics')
        get_info(akw, display=True)
        # add_baseline_stability_estimate(sample) # to try
        if savetoh5:
            subtract_pedestals_all_channels_np(akw, nmax=None, to_h5=True, filename=f'{filepath}/{outputName}.h5')
        else:
            subtract_pedestals_all_channels_np(akw, nmax=None, to_h5=False, filename=f'{filepath}/{outputName}.pkl')

def read_dataset(file, key):
    with h5py.File(file, "r") as h5:
        return h5[key][:]
    
if __name__=="__main__":
    cernbox = '/Users/emiliebertholet/cernbox/coldbox_data'
    filepath = f'{cernbox}/anaCRP_files'

    ####### file by file
    sample = 'cos'

    # if sample=='pns':
    #     runNumber = 25036
    #     # savetoh5 = True
    #     input_files = [
    #         f'{filepath}/ana_pns_r{runNumber}_small_0_7.root', 
    #         f'{filepath}/ana_pns_r{runNumber}_small_8_16.root', 
    #         f'{filepath}/ana_pns_r{runNumber}_small_17_30.root'
    #     ]
    # elif sample=='cos':
    #     runNumber = 25004
    #     # savetoh5 = True
    #     input_files = [
    #         f'{filepath}/ana_cosmic_r{runNumber}_small_0_6.root', 
    #         f'{filepath}/ana_cosmic_r{runNumber}_small_7_15.root', 
    #         f'{filepath}/ana_cosmic_r{runNumber}_small_16_30.root'
    #     ]
    # else:
    #     sys.exit('error')

    # main(sample, runNumber, savetoh5, input_files)

    ####### merge files
    sample = 'pns'
    if sample=='pns':
        runNumber = 25036
        output_name = f'pns_nTuples_r{runNumber}_0-30'
    elif sample=='cos':
        runNumber = 25004
        output_name =f'cos_nTuples_r{runNumber}_0-30'


    files = sorted( glob.glob(os.path.join(filepath, f"{sample}_nTuples_r{runNumber}_*.h5")) )
    pns = {
        "waveformsADC": np.vstack([read_dataset(f, "waveformsADC") for f in files]),
        "pedestals": np.concatenate([read_dataset(f, "pedestals") for f in files])
    }
    with h5py.File(f"{filepath}/{output_name}.h5", "w") as h5:
        for key, arr in pns.items():
            h5.create_dataset(key, data=arr)
    
    


