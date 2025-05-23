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
def waveform_selection(raw_dict, ADCsat=16383, width_range=(100, 500)):
    selection = {
        "peak_indices_2d": [],
        "peak_heights": [],
        "properties": [],
        "waveforms": [],
        "widths": [],
        "pedestals": []
    }

    Ntot, Nsel = 0, 0

    for trigger_idx, (waveform, peak_position, peak_height, props, pedestal) in enumerate(zip(
        raw_dict["waveforms"],
        raw_dict["peak_indices_2d"],
        raw_dict["peak_heights"],
        raw_dict["properties"],
        raw_dict["pedestals"]
    )):
        widths = props.get("widths", [])  ## if the key does not exists, return an empty list
        other_props = {k: v for k, v in props.items()} #if k not in ["widths"]

        # Indices of peaks that satisfy all filters
        keep_waveforms_idx = [
            i for i, (h, w) in enumerate(zip(peak_height, widths))
            if (width_range[0] <= w <= width_range[1]) and (h + pedestal < ADCsat)
        ]

        Ntot += len(peak_position)
        Nsel += len(keep_waveforms_idx)

        if keep_waveforms_idx: # in the case there is no good peak in one trigger event
            selection["waveforms"].append([waveform[i] for i in keep_waveforms_idx])
            selection["peak_indices_2d"].append(np.array([peak_position[i] for i in keep_waveforms_idx]))
            selection["peak_heights"].append(np.array([peak_height[i] for i in keep_waveforms_idx]))
            selection["widths"].append(np.array([widths[i] for i in keep_waveforms_idx]))
            selection["pedestals"].append(pedestal)
            selection["properties"].append({
                **{k: np.array([v[i] for i in keep_waveforms_idx]) for k, v in other_props.items()}
            })

    print(f"Selected {Nsel} peaks out of {Ntot}")

    return selection

def waveform_selection_PNSneutrons(raw_dict, ADCsat=15363, width_range=(100, 500), time_ticks=(3000,6000)):
    selection = {
        "peak_indices_2d": [],
        "peak_heights": [],
        "properties": [],
        "waveforms": [],
        "widths": [],
        "pedestals": []
    }

    Ntot, Nsel = 0, 0

    for trigger_idx, (waveform, peak_position, peak_height, props, pedestal) in enumerate(zip(
        raw_dict["waveforms"],
        raw_dict["peak_indices_2d"],
        raw_dict["peak_heights"],
        raw_dict["properties"],
        raw_dict["pedestals"]
    )):
        widths = props.get("widths", [])  ## if the key does not exists, return an empty list
        other_props = {k: v for k, v in props.items()} #if k not in ["widths"]

        # Indices of peaks that satisfy all filters
        keep_waveforms_idx = [
            i for i, (h, w, peakpos) in enumerate(zip(peak_height, widths, peak_position))
            if (width_range[0] <= w <= width_range[1]) and (h + pedestal < ADCsat) and (time_ticks[0] <= peakpos <= time_ticks[1])
        ]

        Ntot += len(peak_position)
        Nsel += len(keep_waveforms_idx)

        if keep_waveforms_idx: # in the case there is no good peak in one trigger event
            selection["waveforms"].append([waveform[i] for i in keep_waveforms_idx])
            selection["peak_indices_2d"].append(np.array([peak_position[i] for i in keep_waveforms_idx]))
            selection["peak_heights"].append(np.array([peak_height[i] for i in keep_waveforms_idx]))
            selection["widths"].append(np.array([widths[i] for i in keep_waveforms_idx]))
            selection["pedestals"].append(pedestal)
            selection["properties"].append({
                **{k: np.array([v[i] for i in keep_waveforms_idx]) for k, v in other_props.items()}
            })

    print(f"Selected {Nsel} peaks out of {Ntot}")

    return selection

def waveform_selection_PNS_SideBand(raw_dict, ADCsat=15363, width_range=(100, 500), time_ticks=6000):
    selection = {
        "peak_indices_2d": [],
        "peak_heights": [],
        "properties": [],
        "waveforms": [],
        "widths": [],
        "pedestals": []
    }

    Ntot, Nsel = 0, 0

    for trigger_idx, (waveform, peak_position, peak_height, props, pedestal) in enumerate(zip(
        raw_dict["waveforms"],
        raw_dict["peak_indices_2d"],
        raw_dict["peak_heights"],
        raw_dict["properties"],
        raw_dict["pedestals"]
    )):
        widths = props.get("widths", [])  ## if the key does not exists, return an empty list
        other_props = {k: v for k, v in props.items()} #if k not in ["widths"]

        # Indices of peaks that satisfy all filters
        keep_waveforms_idx = [
            i for i, (h, w, peakpos) in enumerate(zip(peak_height, widths, peak_position))
            if (width_range[0] <= w <= width_range[1]) and (h + pedestal < ADCsat) and ( peakpos > time_ticks)
        ]

        Ntot += len(peak_position)
        Nsel += len(keep_waveforms_idx)

        if keep_waveforms_idx: # in the case there is no good peak in one trigger event
            selection["waveforms"].append([waveform[i] for i in keep_waveforms_idx])
            selection["peak_indices_2d"].append(np.array([peak_position[i] for i in keep_waveforms_idx]))
            selection["peak_heights"].append(np.array([peak_height[i] for i in keep_waveforms_idx]))
            selection["widths"].append(np.array([widths[i] for i in keep_waveforms_idx]))
            selection["pedestals"].append(pedestal)
            selection["properties"].append({
                **{k: np.array([v[i] for i in keep_waveforms_idx]) for k, v in other_props.items()}
            })

    print(f"Selected {Nsel} peaks out of {Ntot}")

    return selection


if __name__=="__main__":
    raw_dir = '/Users/emiliebertholet/cernbox/coldbox_data/raw_waveforms'
    sel_dir = '/Users/emiliebertholet/cernbox/coldbox_data/selected_waveforms'

    # ##### cosmics
    filename = 'waveforms_cosmics_wlenNone_prom500.pkl'
    with open(f'{raw_dir}/{filename}', 'rb') as file:
        raw_dict = pickle.load(file)

    sel = waveform_selection(raw_dict, ADCsat=16383, width_range=(100, 500))
    output_file = filename.replace('waveforms', 'selection')
    print(sel.keys())
    # with open(f'{sel_dir}/{output_file}', 'wb') as file:
    # # with open(f'{output_file}', 'wb') as file:
    #     pickle.dump(sel, file)

    ##### PNS
    # filename = 'waveforms_neutrons_wlenNone_prom500.pkl'
    # with open(f'{raw_dir}/{filename}', 'rb') as file:
    #     raw_dict = pickle.load(file)

    # sel = waveform_selection(raw_dict, ADCsat=15363, width_range=(100, 500)) ## ADCsat different for PNS
    # output_file = filename.replace('waveforms_neutrons', 'selection_PNS')
    # with open(f'{sel_dir}/{output_file}', 'wb') as file:
    #     pickle.dump(sel, file)

    # ##### PNS: neutron selection
    # filename = 'waveforms_neutrons_wlenNone_prom500.pkl'
    # with open(f'{raw_dir}/{filename}', 'rb') as file:
    #     raw_dict = pickle.load(file)

    # sel = waveform_selection_PNSneutrons(raw_dict, ADCsat=15363, time_ticks=(3000,6000)) 
    # output_file = filename.replace('waveforms_neutrons', 'selection_PNS_neutrons')
    # with open(f'{sel_dir}/{output_file}', 'wb') as file:
    #     pickle.dump(sel, file)
    
    ##### PNS: side band
    # filename = 'waveforms_neutrons_wlenNone_prom500.pkl'
    # with open(f'{raw_dir}/{filename}', 'rb') as file:
    #     raw_dict = pickle.load(file)

    # sel = waveform_selection_PNS_SideBand(raw_dict, ADCsat=15363, time_ticks=6000) 
    # output_file = filename.replace('waveforms_neutrons', 'selection_PNS_SideBand')
    # with open(f'{sel_dir}/{output_file}', 'wb') as file:
    #     pickle.dump(sel, file)


 


   