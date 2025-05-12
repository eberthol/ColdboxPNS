import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pickle


def substract_peds(row):
    return row - np.argmax(np.bincount(row))

def extract_peaks(data, output_file, window_length=None):
    ped_sub = []
    for row in data:
        ped_sub.append(substract_peds(row))
    ped_sub = np.array(ped_sub)
    ped_sub_flat = ped_sub.flatten()
    peaks, peak_dict = find_peaks(ped_sub_flat, height=None, distance=100, prominence=200, rel_height=0.9, wlen=window_length,  plateau_size=0, width=0 ) 
    print(f'found {len(peaks)} peaks')

    with open(output_file, 'wb') as file:
        pickle.dump(peak_dict, file)


if __name__=="__main__":
    folder_base = '/Users/emiliebertholet/cernbox/coldbox_data'

    # filename = f'{folder_base}/adc_data_run_25036_ch_40_all.npy' # neutrons
    # data = np.load(filename, allow_pickle=True).astype(np.int16)
    # print(filename, data.shape)
    # extract_peaks(data=data, output_file='data_tests/peaks_neutrons_wlenNone.pkl', window_length=None)

    # filename = f'{folder_base}/adc_data_run_25066_ch_40_all.npy' # cosmcis #(4200, 262144)
    # data = np.load(filename, allow_pickle=True).astype(np.int16)
    # print(filename, data.shape)
    # extract_peaks(data=data, output_file='data_tests/peaks_cosmics_wlenNone.pkl', window_length=None)

    # filename = f'{folder_base}/adc_data_run_25036_ch_40_all.npy' # neutrons
    # data = np.load(filename, allow_pickle=True).astype(np.int16)
    # print(filename, data.shape)
    # extract_peaks(data=data, output_file='data_tests/peaks_neutrons_wlen500.pkl', window_length=500)

    # filename = f'{folder_base}/adc_data_run_25066_ch_40_all.npy' # cosmcis #(4200, 262144)
    # data = np.load(filename, allow_pickle=True).astype(np.int16)
    # print(filename, data.shape)
    # extract_peaks(data=data, output_file='data_tests/peaks_cosmics_wlen500.pkl', window_length=500)

    ## neutron region
    filename = f'{folder_base}/adc_data_run_25036_ch_40_all.npy' # neutrons
    data = np.load(filename, allow_pickle=True).astype(np.int16)
    data = data[3000:6000]
    print(filename, data.shape)
    extract_peaks(data=data, output_file='data_tests/peaks_neutronRegion_wlenNone.pkl', window_length=None)

    filename = f'{folder_base}/adc_data_run_25036_ch_40_all.npy' # neutrons
    data = np.load(filename, allow_pickle=True).astype(np.int16)
    data = data[3000:6000]
    print(filename, data.shape)
    extract_peaks(data=data, output_file='data_tests/peaks_neutronRegion_wlen500.pkl', window_length=500)