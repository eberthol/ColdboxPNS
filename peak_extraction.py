import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pickle


'''
input raw data: 2d array
    rows: trigger records (trigRec)
    cols: time ticks (tt)
output format: dictionary (pickle file)
    2d_peak_indices: position of the peaks     [trigRec, tt]
    peak_heights: height of the peak           [trigRec, tt]
    flat_peak_indices: position of the peaks in 1D
    pedestals: one value per trigger record
    waveforms: 
    properties: 

'''
def substract_pedestals(array_2d):
    subtracted = []
    pedestals = []
    for row in array_2d:
        pedestal = np.argmax(np.bincount(row))
        subtracted.append(row - pedestal)
        pedestals.append(pedestal)
    return np.array(subtracted), np.array(pedestals)


def find_peaks_2d(array_2d, **kwargs):
    props_list = []
    peak_indices = []
    peak_heights = []
    flat_peak_indices = []
    peak_slices = []

    for row_idx, row in enumerate(array_2d):
        peaks, props = find_peaks(row, **kwargs)
        heights = props.get("peak_heights", row[peaks])

        # Get slices using left_ips and right_ips (only if width is calculated)
        slices = []
        left_ips = props.get("left_ips")
        right_ips = props.get("right_ips")
        if left_ips is not None and right_ips is not None:
            for l, r in zip(left_ips, right_ips):
                # Convert fractional indices to integers (rounding)
                start = max(0, int(np.floor(l)))
                end = min(len(row), int(np.ceil(r)))
                slices.append(row[start:end])
        else:
            slices = [row[p] for p in peaks]  # fallback: just the peak point

        props_list.append(props)
        peak_indices.append(peaks)
        peak_heights.append(heights)
        flat_peak_indices.extend([(row_idx, col) for col in peaks])
        peak_slices.append(slices)


    return {
        "peak_indices_2d": peak_indices,
        "peak_heights": peak_heights,
        "properties": props_list,
        "waveforms": peak_slices,
        "flat_peak_indices": flat_peak_indices
    }

def save_file(intput_file, output_file, wlen, prominence=200, cut=None):
    raw_data = np.load(intput_file, allow_pickle=True).astype(np.int16)
    data, pedestals = substract_pedestals(raw_data)
    if cut is not None:
            data = data[:, cut[0]:cut[1]]
    print(intput_file, data.shape)
    results_dict = find_peaks_2d(data, height=None, distance=100, 
                                        prominence=prominence, rel_height=0.9, wlen=wlen,  
                                        plateau_size=0, width=0 )
    results_dict["pedestals"] = pedestals 
    with open(output_file, 'wb') as file:
        pickle.dump(results_dict, file)

if __name__=="__main__":
    folder_base = '/Users/emiliebertholet/cernbox/coldbox_data'
    folder_out = '/Users/emiliebertholet/cernbox/coldbox_data/raw_waveforms'

    sample = 'PNS'
    ch, run = '37', '25050'
    intput_file = f'{folder_base}/adc_data_run_{run}_ch_{ch}_all.npy'
    save_file(intput_file=intput_file, 
              output_file=f'{folder_out}/waveforms_{run}ch{ch}_{sample}_wlenNone_prom500.pkl', 
              wlen=None, prominence=500)
    ch, run = '37', '25068'
    intput_file = f'{folder_base}/adc_data_run_{run}_ch_{ch}_all.npy'
    save_file(intput_file=intput_file, 
              output_file=f'{folder_out}/waveforms_{run}ch{ch}_{sample}_wlenNone_prom500.pkl', 
              wlen=None, prominence=500)
    ch, run = '37', '25071'
    intput_file = f'{folder_base}/adc_data_run_{run}_ch_{ch}_all.npy'
    save_file(intput_file=intput_file, 
              output_file=f'{folder_out}/waveforms_{run}ch{ch}_{sample}_wlenNone_prom500.pkl', 
              wlen=None, prominence=500)
    
    sample = 'cosmics'
    ch, run = '37', '25087'
    intput_file = f'{folder_base}/adc_data_run_{run}_ch_{ch}_all.npy'
    save_file(intput_file=intput_file, 
              output_file=f'{folder_out}/waveforms_{run}ch{ch}_{sample}_wlenNone_prom500.pkl', 
              wlen=None, prominence=500)








    ##### neutrons
    # sample = 'neutrons'
    # intput_file = f'{folder_base}/adc_data_run_25036_ch_40_all.npy'

    ##### cosmics
    # sample = 'cosmics'
    # intput_file = f'{folder_base}/adc_data_run_25066_ch_40_all.npy'

    # save_file(intput_file=intput_file, 
    #           output_file=f'data_tests/waveforms_{sample}_wlenNone_prom200.pkl', 
    #           wlen=None, prominence=200)
    
    # save_file(intput_file=intput_file, 
    #           output_file=f'data_tests/waveforms_{sample}_wlen500_prom200.pkl', 
    #           wlen=500, prominence=200)
    
    # save_file(intput_file=intput_file, 
    #           output_file=f'data_tests/waveforms_{sample}_wlenNone_prom500.pkl', 
    #           wlen=None, prominence=500)
    
    # save_file(intput_file=intput_file, 
    #           output_file=f'data_tests/waveforms_{sample}_wlen500_prom500.pkl', 
    #           wlen=500, prominence=500)





    # save_file(intput_file=f'{folder_base}/adc_data_run_25036_ch_40_all.npy', # neutron selection
    #         output_file='data_tests/peaks_neutronsSel_wlenNone.pkl', 
    #         wlen=None, cut=(3000,6000))

    # save_file(intput_file=f'{folder_base}/adc_data_run_25036_ch_40_all.npy', # neutron selection
    #         output_file='data_tests/peaks_neutronsSel_wlen500.pkl', 
    #         wlen=None, cut=(3000,6000))

    
 

    # save_file(intput_file=f'{folder_base}/adc_data_run_25036_ch_40_all.npy', # neutron selection
    #         output_file='data_tests/peaks_neutronsSel_wlenNone_prom500.pkl', 
    #         wlen=None, cut=(3000,6000), prominence=500)
    

    # save_file(intput_file=f'{folder_base}/adc_data_run_25036_ch_40_all.npy', # neutron selection
    #         output_file='data_tests/peaks_neutronsSel_wlen500_prom500.pkl', 
    #         wlen=None, cut=(3000,6000), prominence=500)
    
    


 


   