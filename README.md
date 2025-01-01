
## How git o extract runs

Check the following link: https://docs.google.com/spreadsheets/d/1M0jMwS33TttGbs-REGRAopI9uiohMjcxCvUPJjYtd2g/edit?gid=1816173087#gid=1816173087

To find the run files you wish to extract.

Then, log into a DUNE machine.


### How to log into DUNE machine from windows and extract runs:

install MIT kerberos.

Log into the university VPN.

Then, click "get ticket" in MIT kerberos. Use the following principal: [your_username]@FNAL.GOV (must be uppercase), with the password you received from Fermilab (contact helpdesk for it).

Install putty.

In putty, get to SSH - Auth- GSSAPI, mark all checkboxes.

Log into [your_username]@dunegpvmXX.fnal.gov (replace XX with machine number, I think it goes from 01 to 11).

Runs are in /pnfs/dune/tape_backed/dunepro/vd-coldbox/raw/2024/detector/ For example run 25050 files are in (/pnfs/dune/tape_backed/dunepro/vd-coldbox/raw/2024/detector/cosmics/None/00/02/50/50)

Instructions on how to set the environment are in https://github.com/weishi10141993/VDPDSAna/blob/main/PNSCali/CaliSim.md#generate-gamma-cascades.

use the dump_pds_ana_info.py file from my directory ("ggonen") in the following manner for the desired runs: dump_pds_ana_info.py /pnfs/dune/tape_backed/dunepro/vd-coldbox/raw/2024/detector/cosmics/None/00/02/50/66/ 25066

since the run files are really big, there will be several .npy file generated for each run, divided to chunks (if you run into a memory problem, change the chunck size in dump_pds_ana_info.py). 

Then use winSCP to comfortably transfer all .npy files to your computer, and generate using gpt a simple python code to unite them to one npy file.
