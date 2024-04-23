# SSF implementation 
import numpy as np
from tqdm import tqdm
import json
import time
import datetime
#from bposd import bposd_decoder
from bposd.css import css_code
from itertools import chain,combinations
class ssf_decoding_sim():

    '''
    A class for simulating Small Set Flip decoding of CSS codes

    Note
    ....
    The input parameters can be entered directly or as a dictionary. 

    Parameters
    ----------

    hx: numpy.ndarray
        The hx matrix of the CSS code.
    hz: numpy.ndarray
        The hz matrix of the CSS code.
    seed: int
        The random number generator seed.
    target_runs: int
        The number of runs you wish to simulate.
    output_file: string
        The output file to write to.
    save_interval: int
        The time in interval (in seconds) between writing to the output file.
    check_code: bool
        Check whether the CSS code is valid.
    tqdm_disable: bool
        Enable/disable the tqdm progress bar. If you are running this script on a HPC
        cluster, it is recommend to disable tqdm.
    run_sim: bool
        If enabled (default), the simulation will start automatically.
    '''

    def __init__(self, hx=None, hz=None, **input_dict):

        # default input values
        default_input = {
            'error_rate': None,
            'error_rate': 0,
            'xyz_error_bias': [1, 1, 1],
            'target_runs': 100,
            'seed': 0,
            'output_file': None,
            'save_interval': 2,
            'check_code': 1,
            'tqdm_disable': 0,
            'run_sim': 1
        }

        #apply defaults for keys not passed to the class
        for key in input_dict.keys():
            self.__dict__[key] = input_dict[key]
        for key in default_input.keys():
            if key not in input_dict:
                self.__dict__[key] = default_input[key]

        # output variables
        output_values = {
            "K": None,
            "N": None,
            "start_date": None,
            "runtime": 0.0,
            "runtime_readable": None,
            "run_count": 0,
            "ssf_success": 0,
            "ssf_word_error_rate": 0.0,
            "ssf_logical_error_rate": 0.0,
            "min_logical_weight": 1e9
        }

        for key in output_values.keys(): #copies initial values for output attributes
            if key not in self.__dict__:
                self.__dict__[key] = output_values[key]

        #the attributes we wish to save to file
        temp = [] 
        for key in self.__dict__.keys():
            if key not in ['channel_probs_x','channel_probs_z','channel_probs_y','hx','hz']:
                temp.append(key)
        self.output_keys = temp

        #random number generator setup
        if self.seed==0 or self.run_count!=0:
            self.seed=np.random.randint(low=1,high=2**32-1)
        np.random.seed(self.seed)
        print(f"RNG Seed: {self.seed}")
        
        # the hx and hx matrices
        self.hx = hx.astype(int)
        self.hz = hz.astype(int)
        self.N = self.hz.shape[1] #the block length
        if self.min_logical_weight == 1e9: #the minimum observed weight of a logical operator
            self.min_logical_weight=self.N 
        self.error_x = np.zeros(self.N).astype(int) #x_component error vector
        self.error_z = np.zeros(self.N).astype(int) #z_component error vector

        # construct the CSS code from hx and hz
        self._construct_code()

        # run the small set flip decoder
        #self._ssf()
        # setup the error channel
        self._error_channel_setup()

        # setup the BP+OSD decoders
        self._decoder_setup()

        if self.run_sim:
            self.run_decode_sim()
    def _setofsets(self,universalSet):
        """ Generates the powerset of a list/string/set"""
        # Stackexchanged "How to get all subsest of a set?"
        n = len(universalSet)
        subsets = []
        
        # Generate numbers from 1 to 2^n - 1 (binary representation)
        # Each number corresponds to a subset
        for i in range(1, 2**n):
            subset = []
            for j in range(n):
                # Check if j-th bit of i is set (1)
                if i & (1 << j):
                    subset.append(universalSet[j])
            subsets.append(subset)
        
        return subsets
        #return list(chain.from_iterable(combinations(universalSet,r) for r in range(1,len(universalSet)+1)))



    def _ssf_x(self,synd_x):
        """
        The Small set flip loop . Reference :
        https://theses.hal.science/tel-04018968v1/document
        """
        # This is only working because we have a CSS code.
        # You want to adapt to non CSS Code! 
        # TODO: Treat the full syndrome vector as an atomic unit!! Hx and Hz may require 
        # padding
        # Bit error pair
        Hz = self.hz
        x_error = synd_x.copy()
        min_correct = np.zeros(self.N)
        if sum(synd_x) == 0:
            return min_correct
        # Fix errors by scanning every row in the parity matrix and create a powerset of 
        # non-zero locations . Out of those find the  combination of flips that
        # if done give the largest flip weighted by the size of the flipset.
        #print("Syndrom I see is ",synd_x)
        
        support  = []
        for rows in Hz:
            #Figure out the non-zero locations
            non_zero_idx = []
            for i in range(len(rows)):
                #print("i is ",i)
                if rows[i] == 1:
                    non_zero_idx.append(i)
            #print("nonzero indices are ",len(non_zero_idx))
            # Generate powerset 
            #print("row sis ",rows)
            #print("non zero locations ",non_zero_idx)
            candidates = self._setofsets(non_zero_idx)
            #print(candidates)
            #print("Candidates are ",len(candidates))
            support = support +candidates
        #print(candidates)
        #print("support before redundancy removal ",len(support))
        pre_calc_matrix = []
        weights = []
        errors = []
        for i in range(len(support)):
            probable_error = np.zeros(self.N)
            #print(support[i])
            probable_error[support[i]] = 1
            errors.append(probable_error)
            for positions in support[i]:
                probable_error[positions] = 1
            pre_calc_matrix.append( (self.hz @ probable_error)%2)
            weights.append(sum(probable_error))
        #support = set(support)
        pre_calc_matrix = np.array(pre_calc_matrix)
        weights = np.array(weights)
        #print("self.N is ",self.N)
        #print("precalc matrix is ",weights.shape)
        #print("support after redundancy removal ",len(support))
        #print("after redundancy removal ",len(candidates))
        if sum(x_error)!=0 :
            
            minima = 0
            probable_error = min_correct.copy()
            while(1):
                #print("infinite loop")
                improved = 0
                #print("Improvement perhaps")
                #minima = 0
                 # Flip the syndrome at the locations marked and check the decpreciation in weight
                #probable_error = min_correct
                current_Synd = x_error.copy()
                current_Synd = np.array(current_Synd)
                #print("self.N is ",self.N)
                #probable_error = np.zeros(self.N)
                #print(candidate)
                
                # Measure new syndromw
                new_Syndrome =(current_Synd + pre_calc_matrix) % 2
                #print("new syndrome shape is ",new_Syndrome.shape)
                #if sum(new_Syndrome) == 0:
                #    #print("Error is ",sum(probable_error))
                #    min_correct = (min_correct+probable_error.copy())%2
                #    self.x_inferred_error = min_correct
                #    print("Completely resolved!")
                #    return 
                improvement = (np.sum(current_Synd)-np.sum(new_Syndrome,axis=1))/ weights
                improvement = np.max(improvement)
                #print("improvement is ",improvement)
                loc = np.argmax(improvement)
                if improvement > minima:
                    improved += 1
                    #print("Breakthrough")
                    min_correct_candidate = (min_correct+errors[loc])%2
                    minima = improvement
                    x_error_candidate = new_Syndrome[loc]
                
                if improved == 0:
                    #print("Decoder failure!!")
                    print("min correct weight ",sum(min_correct))
                    print("Success is ",sum((synd_x+self.hz@min_correct)%2))
                    self.x_inferred_error = min_correct
                    return   
                x_error = x_error_candidate.copy()    
                min_correct = min_correct_candidate
        #self.x_inferred_error = min_correct

    def _ssf_z(self,synd_z):
        """
        The Small set flip loop . Reference :
        https://theses.hal.science/tel-04018968v1/document
        """
        # This is only working because we have a CSS code.
        # You want to adapt to non CSS Code! 
        # TODO: Treat the full syndrome vector as an atomic unit!! Hx and Hz may require 
        # padding
        # Bit error pair
        Hx = self.hx
        #print("Hx is ",Hx)
        z_error = synd_z
        min_correct = np.zeros(self.N)
        #print("input error is ",sum(z_error))
        if sum(z_error) == 0:
            return min_correct
        # Fix errors by scanning every row in the parity matrix and create a powerset of 
        # non-zero locations . Out of those find the  combination of flips that
        # if done give the largest flip weighted by the size of the flipset.
        if sum(z_error)!=0 :
            
            
            
            while(1):
                improved = 0
                #print("Improvement perhaps")
                minima = 0
                #print("Here")
                # Find the non zero indicex in the syndrome chain
                bad_index = []
                for j in range(len(z_error)):
                    if z_error[j]==1:
                        bad_index.append(j)
                #print("bad indexes ",bad_index)
                for index in bad_index:
                    rows = Hx[index]
                    #print("rows time seen ",improved)
                    #Figure out the non-zero locations
                    non_zero_idx = []
                    for i in range(len(rows)):
                        #print("i is ",i)
                        if rows[i] == 1:
                            non_zero_idx.append(i)
                    # Generate powerset 
                    #print("\n row sis ",rows[:10]," \n")
                    #print(" \n non zero locations ",non_zero_idx)    
                    candidates = self._setofsets(non_zero_idx)
                    #print("total length is ",len(candidates))    
                    
                    for candidate in candidates:
                        # Flip the syndrome at the locations marked and check the decpreciation in weight
                        probable_error = min_correct
                        current_Synd = z_error
                        for positions in candidate:
                            probable_error[positions] = (probable_error[positions]+1) % 2
                        # Measure new syndromw
                        new_Syndrome = (self.hx @ probable_error) % 2
                        #print("\n new syndrome weights ",sum(new_Syndrome))
                        #if sum(new_Syndrome) == 0:
                        #    self.z_inferred_error = probable_error
                        #    #print("Completely resolved!")
                        #    return 
                        improvement = (sum(current_Synd)-sum(new_Syndrome))/ len(candidate)
                        #if sum(new_Syndrome) == 0:
                            #print("Cndidate length is ",len(candidate))
                        if improvement > minima:
                            improved += 1
                            #print("Breakthrough ",sum(new_Syndrome))
                            min_correct_candidate = probable_error
                            minima = improvement
                            z_error_candidate = new_Syndrome
                           
                if improved == 0:
                    #print("Decoder failure!! ", sum(z_error))
                    self.z_inferred_error = min_correct
                    return    
                z_error = z_error_candidate   
                min_correct = (min_correct+min_correct_candidate)%2   
                if sum(z_error) == 0:
                    #print("Solved")
                    self.z_inferred_error = min_correct
                    #print("Seems like our error is ",sum((self.error_z +min_correct)%2))
                    return 
                         


        

    def _single_run(self):

        '''
        The main simulation procedure
        '''
        # Parity matrices
        #Bit pair
        # TODO: Paste both errors in one .Replace with full ssf call
        # randomly generate the error
        self.error_x, self.error_z = self._generate_error()
        self.x_inferred_error = np.zeros(self.N)
        self.z_inferred_error = np.zeros(self.N)
        synd_z = self.hx@self.error_z % 2
        self._ssf_z(synd_z)
        synd_x = self.hz@self.error_x % 2
        self._ssf_x(synd_x)

        #compute the logical and word error rates
        self._encoded_error_rates()

    



    def _encoded_error_rates(self):

        '''
        Updates the logical and word error rates for SSF
        '''

        #OSDW Logical error rate
        # calculate the residual error
        residual_x = (self.error_x+self.x_inferred_error) % 2
        residual_z = (self.error_z+self.z_inferred_error) % 2

        # check for logical X-error
        if (self.lz@residual_x % 2).any():
            logical_weight = np.sum(residual_x)
            if logical_weight < self.min_logical_weight:
                print("Here")
                self.min_logical_weight = int(logical_weight)

        # check for logical Z-error
        elif (self.lx@residual_z % 2).any():
            #print("Error happened")
            logical_weight = np.sum(residual_z)
            if logical_weight < self.min_logical_weight:
                self.min_logical_weight = int(logical_weight)
        else:
            self.ssf_success += 1

        # compute logical error rate
        self.ssf_logical_error_rate = 1-self.ssf_success/self.run_count
        self.ssf_logical_error_rate_eb = np.sqrt(
            (1-self.ssf_logical_error_rate)*self.ssf_logical_error_rate/self.run_count)
        self.ssf_word_error_rate_eb = self.ssf_logical_error_rate_eb * \
            ((1-self.ssf_logical_error_rate_eb)**(1/self.K - 1))/self.K

        # compute word error rate
        self.ssf_word_error_rate = 1.0 - \
            (1-self.ssf_logical_error_rate)**(1/self.K)
        

    def _construct_code(self):

        '''
        Constructs the CSS code from the hx and hz stabilizer matrices. [SAME]
        '''

        print("Constructing CSS code from hx and hz matrices...")
        if isinstance(self.hx, np.ndarray) and isinstance(self.hz, np.ndarray):
            qcode = css_code(self.hx, self.hz)
            self.lx = qcode.lx
            self.lz = qcode.lz
            self.K = qcode.K
            self.N = qcode.N
            print("Checking the CSS code is valid...")
            if self.check_code and not qcode.test():
                raise Exception(
                    "Error: invalid CSS code. Check the form of your hx and hz matrices!")
        else:
            raise Exception("Invalid object type for the hx/hz matrices")
        return None

    def _error_channel_setup(self):

        '''
        Sets up the error channels from the error rate and error bias input parameters
        '''

        xyz_error_bias = np.array(self.xyz_error_bias)
        if xyz_error_bias[0] == np.inf:
            self.px = self.error_rate
            self.py = 0
            self.pz = 0
        elif xyz_error_bias[1] == np.inf:
            self.px = 0
            self.py = self.error_rate
            self.pz = 0
        elif xyz_error_bias[2] == np.inf:
            self.px = 0
            self.py = 0
            self.pz = self.error_rate
        else:
            self.px, self.py, self.pz = self.error_rate * \
                xyz_error_bias/np.sum(xyz_error_bias)
        print("px,py,pz ",self.px,self.py,self.pz)
        
        self.channel_probs_x = np.ones(self.N)*(self.px)
        self.channel_probs_z = np.ones(self.N)*(self.pz)
        self.channel_probs_y = np.ones(self.N)*(self.py)

        self.channel_probs_x.setflags(write=False)
        self.channel_probs_y.setflags(write=False)
        self.channel_probs_z.setflags(write=False)


    def _decoder_setup(self):

        '''
        Setup for the SSF decoders [IGNORE]
        '''
        pass
       

    def _generate_error(self):

        '''
        Generates a random error on both the X and Z components of the code
        distributed according to the channel probability vectors.
        '''
        #TODO : Patch the errors up in one unit . 
        for i in range(self.N):
            rand = np.random.random()
            if rand < self.channel_probs_z[i]:
                self.error_z[i] = 1
                self.error_x[i] = 0
            elif self.channel_probs_z[i] <= rand < (self.channel_probs_z[i]+self.channel_probs_x[i]):
                self.error_z[i] = 0
                self.error_x[i] = 1
            elif (self.channel_probs_z[i]+self.channel_probs_x[i]) <= rand < (self.channel_probs_x[i]+self.channel_probs_y[i]+self.channel_probs_z[i]):
                self.error_z[i] = 1
                self.error_x[i] = 1
            else:
                self.error_z[i] = 0
                self.error_x[i] = 0

        return self.error_x, self.error_z
 
    def run_decode_sim(self):

        '''
        This function contains the main simulation loop and controls the output.
        '''

        # save start date
        self.start_date = datetime.datetime.fromtimestamp(
            time.time()).strftime("%A, %B %d, %Y %H:%M:%S")

        pbar = tqdm(range(self.run_count+1, self.target_runs+1),
                    disable=self.tqdm_disable, ncols=0)

        start_time = time.time()
        save_time = start_time

        for self.run_count in pbar:

            self._single_run()

            pbar.set_description(f"d_max: {self.min_logical_weight}; SSF_WER: {self.ssf_word_error_rate*100:.3g}±{self.ssf_word_error_rate_eb*100:.2g}%; SSF_logical: {self.ssf_logical_error_rate*100:.3g}±{self.ssf_logical_error_rate_eb*100:.2g}%; ")

            current_time = time.time()
            save_loop = current_time-save_time

            if int(save_loop)>self.save_interval or self.run_count==self.target_runs:
                save_time=time.time()
                self.runtime = save_loop +self.runtime

                self.runtime_readable=time.strftime('%H:%M:%S', time.gmtime(self.runtime))


                if self.output_file!=None:
                    f=open(self.output_file,"w+")
                    print(self.output_dict(),file=f)
                    f.close()

                #if self.osdw_logical_error_rate_eb>0 and self.osdw_logical_error_rate_eb/self.osdw_logical_error_rate < self.error_bar_precision_cutoff:
                #    print("\nTarget error bar precision reached. Stopping simulation...")
                #    break

        return json.dumps(self.output_dict(),sort_keys=True, indent=4)

    def output_dict(self):

        '''
        Function for formatting the output
        '''

        output_dict = {}
        for key, value in self.__dict__.items():
            if key in self.output_keys:
                output_dict[key] = value
        # return output_dict
        return json.dumps(output_dict,sort_keys=True, indent=4)

# Example usage


