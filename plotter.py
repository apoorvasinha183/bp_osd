import json
import numpy as np
import matplotlib.pyplot as plt

# Load all the related files
M = np.array([16,20,24,28])
error = [0.005,0.008,0.01,0.02,0.05,0.08,0.1]
best_code = None
for n_qbits in M:
    error_f = []
    for error_rate in error:
    
        n_check = int(n_qbits*3/4)
        #dv = 3
        #dc = 4
        #max_distance = 0
        
        fname = "montecarloresults//bposd_final//"+"nq"+str(n_qbits)+"nc"+str(n_check)+"random"+"error"+str( int(error_rate*1000))+".json"
        f = open(fname)
        data = json.load(f)
        #print("Data keys are ",data.keys())
        #for keys in data:
        #    print(keys)
        error_f.append(data["osdw_word_error_rate"])
    # Brute Force
    k = 0
    d = 0
    if n_qbits == 16:
        N = 400
        k = 16
        d = 8
    if n_qbits == 20:
        N = 625
        k = 25
        d = 8    
    if n_qbits == 24:
        N = 900
        k = 36
        d = 10 
    if n_qbits == 28:
        N = 1225
        k = 49
        d = 12    
        
    name = "[["+str(N)+","+str(k)+","+str(d)+"]]"
    error_eff = error.copy()
    error_eff_out = error_f.copy()
    if n_qbits==24:
        error_eff = error_eff[2:]
        error_eff_out = error_eff_out[2:]
    if n_qbits==28:
        error_eff = error_eff[3:]
        error_eff_out = error_eff_out[3:]

    #print("printing ",error_eff)    
    plt.plot(error_eff,error_eff_out,label=name)
plt.plot(error,error,label ='Threshold Line ',linestyle='dashed')    
# distance 10 ad 12 need help
plt.xlabel('Input Physical Error Rate')
plt.ylabel('Logical/Word error rate')
plt.ylim(1e-7,0.2)

plt.legend(fontsize='large')   # Set the font size of the legend
plt.legend(title='Legend')
plt.yscale('log')
plt.xscale('log')
plt.legend(loc='lower right')  
plt.savefig("FinalResults.png")
plt.title("Hypergraph Product Codes Plot")
plt.show()    

