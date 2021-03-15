#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 12:12:55 2019

@author: N. Youssef 

Starting at a random amino acid sequence, evolve protein until sequences with high fitness(>0.99) 
given the specified native structure (and set of decoy structures) is obtained

"""
import misc_functions as mf
import stability_functions as sf
import pickle
import numpy as np
import random
import argparse
import os

parser = argparse.ArgumentParser(description='obtain sequences with fitness > 0.99')
parser.add_argument('-p', '--protein', help = 'Protein name',required=True)
parser.add_argument('-t', '--trial',   help = 'Trial number', required=True)
parser.add_argument('-o', '--output_sequence',   help = 'specifies if output sequence should be amino acids or codons', required=True)

args = parser.parse_args()

protein = str(args.protein)
trial = int(args.trial)
aa_or_cdns = str(args.output_sequence)

num_cores = 10
#num_cores = int(os.environ.get('SLURM_CPUS_PER_TASK',default=1))

path_to_contact_maps = '../contact_maps/'

def modified_site_specific_fit(seq, full_ssF):
    '''
        Assigns the resident amino acids a fitness value of -1 to ensure a substitution occurs
    '''    
    current_fit = full_ssF[0, seq[0]] # the current sequence fitness can be calculated from the ssFitnesses as the fitness of the resident amino acid at any of the sites. Here site = 0
    for site in range(len(seq)):
        for aa in range(20):
            if seq[site] == aa:
                full_ssF[site, aa] = -1
    return(full_ssF, current_fit)

def higher_fitness(full_ssF, current_fit):
    '''
        returns a site and aa change that would increase fitness 
        Note: this need NOT be the *best* subsitutions that can lead to local sub optimal traps
    '''
    idx  =  np.random.randint(len(np.where(full_ssF > current_fit)[0]))
    site =  np.where(full_ssF > current_fit)[0][idx]
    aa   =  np.where(full_ssF > current_fit)[1][idx]
    return(site, aa)
    
#Load native state contact map˜
with open(path_to_contact_maps  + protein + "_DICT.txt", "rb") as file:
    ns_contact_map = pickle.load(file)
    num_sites = len(ns_contact_map)

#load alternative state contact map
with open(path_to_contact_maps + "/ALT_DICT.txt", "rb") as file:
    alt_contact_maps = pickle.load(file)


#%%
def find_high_fit_seq(seq, current_fit): 
    while current_fit < 0.99:
        
        full_ssF = sf.full_ss_fitness(seq, num_cores, ns_contact_map, alt_contact_maps)[0]
        modified_full_ssF, current_fit = modified_site_specific_fit(seq, full_ssF)
    
    
        if np.max(modified_full_ssF) < current_fit:
        # if none of the single step amino acid changes will increase fitness
        # then randomly choose 20 sites and change them to higher fit aa
        
            sites = np.random.choice(len(seq), 20)
            new_seq = list(seq)
            for site in sites:
                aa = np.where(modified_full_ssF[site] == modified_full_ssF[site].max())[0][0]
                new_seq[site] = int(aa)
            seq = list(new_seq)        
        
        else:
            site, aa = higher_fitness(modified_full_ssF, current_fit)
            new_seq = list(seq)
            new_seq[site] = int(aa)
            seq = list(new_seq)
            
        print(mf.aa_index_to_aa(seq), flush = True)
        print(str(current_fit) + '\n\n', flush = True)
        print(site, aa, flush = True)
#            if current_fit > max_fit:
#                good_seq = list(seq)
#                max_fit = current_fit
    return(seq)

    
#%% 
# Randomly assign initial sequence
initial_seq = random.choices([x for x in range(20)], k =num_sites)
seq = list(initial_seq)
    
max_fit = -0.01 #keep track of maximum fitness 
    
full_ssF = sf.full_ss_fitness(seq, num_cores, ns_contact_map, alt_contact_maps)[0]
modified_full_ssF, current_fit = modified_site_specific_fit(seq, full_ssF)


if aa_or_cdns == 'aa':
    with open("../data/eq_seqs/"+ protein +"_equilibrium_seq_t"+ str(trial)+".txt", "a+") as file:
        eq_seq = find_high_fit_seq(seq, current_fit)
        aa_seq  = mf.aa_index_to_aa(eq_seq)
        file.write(aa_seq + "\n" )

elif aa_or_cdns == 'cdn':
    with open("../data/eq_seqs/"+ protein +"_equilibrium_cdn_seq_t"+ str(trial)+".txt", "a+") as file:
        eq_seq  = find_high_fit_seq(seq, current_fit)
        aa_seq  = mf.aa_index_to_aa(eq_seq)
        cdn_seq = mf.aa_to_cdn_seq(aa_seq)
        file.write(cdn_seq + "\n" )

