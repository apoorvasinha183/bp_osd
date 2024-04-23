import numpy as np
from bposd.hgp import hgp
from bposd.css_decode_sim import css_decode_sim
import os
path = "examples//codes//classical_seed_codes"
dir_list = os.listdir(path)
print("Files and directories in '", path, "' :")
# prints all files
print(dir_list)
nruns = 10
error = [0.008]
for error_rate in error:
    for files in dir_list:
        fn = path+"/"+files
        print(fn)
        h=np.loadtxt(fn).astype(int)
        qcode=hgp(h,compute_distance=True) # construct quantum LDPC code using the symmetric hypergraph product
        qcode.print_code_parameters()
        fname = "montecarloresults//bposd//ignore"+files[:11]+"error"+str( int(error_rate*100))+".json"
        #print(fname)
        osd_options={
        'error_rate': error_rate,
        'target_runs': nruns,
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

        lk = css_decode_sim(hx=qcode.hx, hz=qcode.hz, **osd_options)

