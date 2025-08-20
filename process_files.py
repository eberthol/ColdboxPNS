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

def estimate_baseline_stability(waveform, max_deviation=5):
    median = np.median(waveform)
    mad    = np.median(np.abs(waveform - median))

    quiet = np.abs(waveform - median) < max_deviation * mad

    if not np.any(quiet):  # if mask is empty
        return np.nan, np.nan, quiet ### maybe should return an arbitrary high number instead (e.g. 99999)

    baseline = np.mean(waveform[quiet])
    rms = np.std(waveform[quiet])
    return baseline, rms, quiet

def subtract_pedestals_all_channels_np(akw):
    waveforms = akw['waveformsADC']
    nevents   = len(waveforms)
    waveforms = waveforms[:nevents]

    # Determine number of channels and maximum waveform length
    n_channels = max(len(evt) for evt in waveforms)
    max_len    = max(len(ch) for evt in waveforms for ch in evt)

    output = {}
    for chID in range(n_channels):
        waveformsADC_np = np.full((nevents, max_len), np.nan, dtype=np.int32)
        pedestals_np = np.zeros(nevents, dtype=np.int32)
        baseline_stability_np_ch = np.full(nevents, np.nan, dtype=np.float32)

        for evtID, evt in enumerate(tqdm(waveforms, desc=f"Subtracting pedestals ch{chID}")):
            ch_np = np.asarray(evt[chID], dtype=np.int32)

            # pedestal subtraction
            hist = np.bincount(ch_np)
            pedestal = np.argmax(hist)
            pedestals_np[evtID] = pedestal
            sub = ch_np - pedestal
            waveformsADC_np[evtID, :len(ch_np)] = sub  

            # baseline stability estimate
            _, rms, _ = estimate_baseline_stability(sub)
            baseline_stability_np_ch[evtID] = rms

        output[f"ch{chID}"] = {
                "waveforms": waveformsADC_np, ## pedestal subtracted waveforms
                "pedestals": pedestals_np,
                "baseline_stability": baseline_stability_np_ch
            }
    return output

def main(sample, runNumber, n_channels, input_files, output_directory):
    for filepath in input_files:
        print('process file', filepath)
        info=filepath.split('/')[-1].split('.')[0].split('_')
        id_first, id_last = info[-2], info[-1]
        outputName = f'{sample}_nTuples_r{runNumber}_{id_first}-{id_last}.root'
        if sample=='pns':
            akw = read_root_files(filepath, cosmics=False)
            print('\nPNS')
        else:
            akw = read_root_files(filepath, cosmics=True)
            print('\nCosmics')
        get_info(akw, display=True)

        output_dict = subtract_pedestals_all_channels_np(akw)

        with uproot.recreate(f"{output_directory}/{outputName}") as file:
            for i in range(n_channels):
                file[f"ch{i}"] = output_dict[f"ch{i}"]
        
def read_dataset(file, key):
    with h5py.File(file, "r") as h5:
        return h5[key][:]
    
if __name__=="__main__":
    cernbox = '/Users/emiliebertholet/cernbox/coldbox_data'
    filepath = f'{cernbox}/anaCRP_files/raw_files'
    output_directory = f'{cernbox}/waveform_nTuples'

    ####### file by file
    sample = 'pns'

    if sample=='pns':
        runNumber = 25036
        # input_files = [ f'{filepath}/{sample}_r{runNumber}/ana_pns_small_{i*10}_{i*10+9}.root' for i in range(5, 22)]
        input_files = [ f'{filepath}/{sample}_r{runNumber}/ana_pns_small_{i*10}_{i*10+9}.root' for i in range(6, 10)]



    elif sample=='cos':
        runNumber = 25004
        # input_files = [ f'{filepath}/{sample}_r{runNumber}/ana_cos_small_{i*10}_{i*10+9}.root' for i in range(5, 23)]
        input_files = [ f'{filepath}/{sample}_r{runNumber}/ana_cos_small_{i*10}_{i*10+9}.root' for i in range(6, 10)]

    else:
        sys.exit('error')

    n_channels = 12
    # print(input_files)
    main(sample, runNumber,  n_channels, input_files, output_directory)

    ####### merge files
    # sample = 'pns'
    # if sample=='pns':
    #     runNumber = 25036
    #     output_name = f'pns_nTuples_r{runNumber}_0-30'
    # elif sample=='cos':
    #     runNumber = 25004
    #     output_name =f'cos_nTuples_r{runNumber}_0-30'


    # files = sorted( glob.glob(os.path.join(filepath, f"{sample}_nTuples_r{runNumber}_*.h5")) )
    # pns = {
    #     "waveformsADC": np.vstack([read_dataset(f, "waveformsADC") for f in files]),
    #     "pedestals":    np.concatenate([read_dataset(f, "pedestals") for f in files])
    # }
    # with h5py.File(f"{filepath}/{output_name}.h5", "w") as h5:
    #     for key, arr in pns.items():
    #         h5.create_dataset(key, data=arr)
    
    


