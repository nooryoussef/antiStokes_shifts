#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 2020
@author: nooryoussef

Calculate shannon entropy of propensity landscapes
"""
import numpy as np
import pandas as pd
from scipy import stats 

protein  = '1qhw'
start_trial = 1
end_trial = 500

for trial in range(start_trial, end_trial + 1):
    print(trial, flush = True)
    ss_freq = np.load("../output/" + protein + "/" + protein + '_Ne2_t' + str(trial) + '_ssfreq.npy') 
    num_subs, num_sites = ss_freq.shape[0:2]
    
    uni_idx  = stats.entropy(ss_freq, axis = 2)
    
    #write to output file 
    np.savetxt("../data/uni_idx/" + protein + "/" + protein + '_Ne2_t' + str(trial) + "_uni_idx.csv", uni_idx)
