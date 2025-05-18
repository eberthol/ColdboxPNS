import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pickle


def substract_peds(row):
    return row - np.argmax(np.bincount(row))

def find_peaks_2d(array_2d, **kwargs):
    masks = []
    props_list = []
    peak_indices = []
    flat_peak_indices = []

    for row_idx, row in enumerate(array_2d):
        peaks, props = find_peaks(row, **kwargs)
        mask = np.isin(np.arange(row.size), peaks)
        masks.append(mask)
        props_list.append(props)
        peak_indices.append(peaks)
        flat_peak_indices.extend([(row_idx, col) for col in peaks])

    peak_masks = np.array(masks)

    return {
        "mask": peak_masks,
        "row_peak_indices": peak_indices,
        "properties": props_list,
        "flat_peak_indices": flat_peak_indices
    }

def save_file(intput_file, output_file, wlen, prominence=200, cut=None):
        data = np.load(intput_file, allow_pickle=True).astype(np.int16)
        data = np.apply_along_axis(substract_peds, axis=1, arr=data)
        if cut is not None:
             data = data[:, cut[0]:cut[1]]
        print(intput_file, data.shape)
        results_dict = find_peaks_2d(data, height=None, distance=100, 
                                           prominence=prominence, rel_height=0.9, wlen=wlen,  
                                           plateau_size=0, width=0 )
        with open(output_file, 'wb') as file:
            pickle.dump(results_dict, file)


if __name__=="__main__":
    folder_base = '/Users/emiliebertholet/cernbox/coldbox_data'

    
    # save_file(intput_file=f'{folder_base}/adc_data_run_25036_ch_40_all.npy', # neutrons
    #           output_file='data_tests/peaks_neutrons_wlenNone.pkl', 
    #           wlen=None)

    # save_file(intput_file=f'{folder_base}/adc_data_run_25036_ch_40_all.npy', # neutron selection
    #         output_file='data_tests/peaks_neutronsSel_wlenNone.pkl', 
    #         wlen=None, cut=(3000,6000))
    
    # save_file(intput_file=f'{folder_base}/adc_data_run_25066_ch_40_all.npy', # cosmics
    #           output_file='data_tests/peaks_cosmics_wlenNone.pkl', 
    #           wlen=None)

    # save_file(intput_file=f'{folder_base}/adc_data_run_25036_ch_40_all.npy', # neutrons
    #           output_file='data_tests/peaks_neutrons_wlen500.pkl', 
    #           wlen=500)

    # save_file(intput_file=f'{folder_base}/adc_data_run_25036_ch_40_all.npy', # neutron selection
    #         output_file='data_tests/peaks_neutronsSel_wlen500.pkl', 
    #         wlen=None, cut=(3000,6000))
    
    # save_file(intput_file=f'{folder_base}/adc_data_run_25066_ch_40_all.npy', # cosmics
    #           output_file='data_tests/peaks_cosmics_wlen500.pkl', 
    #           wlen=500)




    save_file(intput_file=f'{folder_base}/adc_data_run_25036_ch_40_all.npy', # neutrons
              output_file='data_tests/peaks_neutrons_wlenNone_prom500.pkl', 
              wlen=None, prominence=500)

    save_file(intput_file=f'{folder_base}/adc_data_run_25036_ch_40_all.npy', # neutron selection
            output_file='data_tests/peaks_neutronsSel_wlenNone_prom500.pkl', 
            wlen=None, cut=(3000,6000), prominence=500)
    
    save_file(intput_file=f'{folder_base}/adc_data_run_25066_ch_40_all.npy', # cosmics
              output_file='data_tests/peaks_cosmics_wlenNone_prom500.pkl', 
              wlen=None, prominence=500)

    save_file(intput_file=f'{folder_base}/adc_data_run_25036_ch_40_all.npy', # neutrons
              output_file='data_tests/peaks_neutrons_wlen500_prom500.pkl', 
              wlen=500, prominence=500)

    save_file(intput_file=f'{folder_base}/adc_data_run_25036_ch_40_all.npy', # neutron selection
            output_file='data_tests/peaks_neutronsSel_wlen500_prom500.pkl', 
            wlen=None, cut=(3000,6000), prominence=500)
    
    save_file(intput_file=f'{folder_base}/adc_data_run_25066_ch_40_all.npy', # cosmics
              output_file='data_tests/peaks_cosmics_wlen500_prom500.pkl', 
              wlen=500, prominence=500)


 


   