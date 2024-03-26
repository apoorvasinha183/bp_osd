import sys
sys.path.append("src")
import numpy as np
from bposd.hgp import hgp
#from bposd.ssf_decoding_sim import ssf_decode_sim
from src.bposd_copy.ssf_decoding_sim import ssf_decoding_sim
import os
path = "examples//codes//classical_seed_codes"
dir_list = os.listdir(path)
print("Files and directories in '", path, "' :")
# prints all files
print(dir_list)
nruns = 10000
error = [0.01,0.05,0.01]
for error_rate in error:
    for files in dir_list:
        fn = path+"/"+files
        print(fn)
        h=np.loadtxt(fn).astype(int)
        qcode=hgp(h) # construct quantum LDPC code using the symmetric hypergraph product
        fname = "montecarloresults//ssf//"+files[:11]+"error"+str( int(error_rate*100))+".json"
        #print(fname)
        osd_options={
        'error_rate': error_rate,
        'target_runs': 10000,
        'xyz_error_bias': [0, 0, 1],
        'output_file': fname,
        'bp_method': "ms",
        'ms_scaling_factor': 0,
        'osd_method': "osd_cs",
        'osd_order': 42,
        'channel_update': None,
        'seed': 42,
        'max_iter': 0,
        'output_file': fname
        }

        lk = ssf_decoding_sim(hx=qcode.hx, hz=qcode.hz, **osd_options)

