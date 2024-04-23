import numpy as np
from bposd.hgp import hgp
from bposd.css_decode_sim import css_decode_sim
import networkx as nx
import matplotlib.pyplot as plt
#from bposd.ssf_decoding_sim import ssf_decode_sim
from src.bposd_copy.ssf_decoding_sim import ssf_decoding_sim
from src.bposd_copy.random_seed import generate_biregular_graph
from src.bposd_copy.random_seed import bipartite_to_adjacency_matrix
import os
nruns = 100000 # Dynamic Adjustment needed
M = np.array([44])
N = M*4
print(N)
error = [0.005,0.008,0.01,0.02,0.05,0.08,0.1]
best_code = None
for error_rate in error:
    for n_qbits in M:
        n_check = int(n_qbits*3/4)
        dv = 3
        dc = 4
        max_distance = 0
        if best_code is None:
            for _ in range(200):
                max_attempts = 10000000
                graph = generate_biregular_graph(n_qbits, n_check, dv, dc, max_attempts=max_attempts)
                h = bipartite_to_adjacency_matrix(graph,n_qbits)
                #print(h)
            #fn = path+"/"+files
            #print(fn)
            #h=np.loadtxt(fn).astype(int)
                qcode=hgp(h1=h,h2=h,compute_distance=True) # construct quantum LDPC code using the symmetric hypergraph product
                #qcode.print_code_parameters()
                distance = qcode.D
                if distance > max_distance:
                    max_distance = distance
                    best_code = qcode
                print("Best so far is ",best_code.print_code_parameters())
        qcode = best_code
        print("Best candidate has ",qcode.D)
        fname = "montecarloresults//bposd_final//"+"nq"+str(n_qbits)+"nc"+str(n_check)+"random"+"error"+str( int(error_rate*1000))+".json"
        #print(fname)
        #Only Z-errors 
        if error_rate > 0.02:
            nruns = 1000 # BP can take forever 
        osd_options={
        'error_rate': error_rate,
        'target_runs': nruns,
        'xyz_error_bias': [0, 0, 1], #Z only This is CSS
        'output_file': fname,
        'bp_method': "ms",
        'ms_scaling_factor': 0,
        'osd_method': "osd_cs",
        'osd_order': 42,   # You can increase it . But 42 is enoguh.
        'channel_update': None,
        'seed': 42,
        'max_iter': 0,
        'output_file': fname
        }

        lk = css_decode_sim(hx=qcode.hx, hz=qcode.hz, **osd_options)

