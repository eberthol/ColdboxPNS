SPE Template and Deconvolution Analysis

This repository contains code for producing Single Photoelectron (SPE) templates and performing SPE deconvolution and pulse-shape discrimination (PSD) studies using PNS and cosmic data from the DUNE Vertical Drift ColdBox.

1. Producing SPE Templates

ROOT macro: AnalyzercodeCT_feb24.C

The macro reads LED calibration waveforms and produces SPE response templates for the photodetector channels.

To run:

root -l AnalyzercodeCT_feb24.C

Input LED calibration files for the C4 module are located in the C4/ directory.

The macro selects SPE pulses and constructs an average SPE response template.

The resulting SPE templates are saved as ROOT files.

For the C4 module, the templates currently used are:

C4_AFE3_ch0_SPE.root

C4_AFE3_ch7_SPE.root

The SPE template is stored in the histogram SPEtemplate_ns.

2. SPE Deconvolution

Jupyter notebook: spe_deconvolution.ipynb

Input .root file can found:
--------------------------------------------------------------
/exp/dune/data/users/rlalnunt/vdcoldcox/ana_pns/cosmic_*.root
---------------------------------------------------------------
This notebook reads the SPE templates and applies Wiener-like deconvolution to optical waveforms from PNS and cosmic datasets.

The analysis includes:

Loading SPE templates: Read the SPE response histograms from ROOT files and prepare them for deconvolution.

C4 coincidence selection: Select coincident optical signals from channels 4 and 7 within an 80 ns time window, applying reconstructed PE requirements.

Waveform preprocessing: Extract the ADC waveform around each selected pulse and subtract the baseline.

SPE deconvolution: Apply frequency-domain regularized deconvolution using the corresponding SPE response template.

Waveform reconstruction: Convolve the recovered signal with the SPE template and compare it with the measured ADC waveform.

Pulse-shape discrimination: Calculate the prompt-light fraction:


[
F_{\mathrm{prompt}}=\frac{Q_{\mathrm{prompt}}}{Q_{\mathrm{total}}}
]

PNS and cosmic comparison: Compare PSD distributions for channels 4 and 7 and study their dependence on total recovered charge and additional waveform peaks.

The deconvolution and PSD calculations are performed separately for each channel after the C4 coincidence selection.
