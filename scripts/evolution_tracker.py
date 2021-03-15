#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 15:54:31 2020

@author: nooryoussef

Starting at an equilibriated sequence, evolve protein while keeping track of 
fitness, dG, site-specific fitness landscape at all sites, and dG landscape at all sites

"""
import evolve_functions as ef 
import misc_functions as mf 
import stability_functions as sf
import pickle
import numpy as np 
import argparse
import os

###############################################################
#            Simulation Parameters        
###############################################################
parser = argparse.ArgumentParser(description='Evolve protein while keeping track of site-specific fitness landscape and dG at all sites')
parser.add_argument('-p', '--protein', help = 'Protein name',required=True)
parser.add_argument('-t', '--trial',   help = 'Trial number', required=True)
parser.add_argument('-n', '--Ne',      help = 'Effective population size', required=True)
parser.add_argument('-s', '--sub',     help = 'Number of substitutions', required=True)
args = parser.parse_args()

protein = str(args.protein)
trial = int(args.trial)
Neff = int(args.Ne)
num_subs = int(args.sub)
NE = 'Ne' + str(len(str(Neff)) - len(str(Neff).rstrip('0')))

num_cores = 20
#num_cores = int(os.environ.get('SLURM_CPUS_PER_TASK',default=1))

#Load native state contact map˜
with open("../contact_maps/" + protein + "_DICT.txt", "rb") as file:
    ns_contact_map = pickle.load(file)

#Load alternative state contact map
with open("../contact_maps/" + "ALT_DICT.txt", "rb") as file:
    alt_contact_maps = pickle.load(file)
    
    
### protein-specific mutation parameters ### 
if protein == '1qhw':
    #1qhw
    pi_nuc = [0.19675, 0.31761, 0.28032, 0.20532] #['T' ,'C' ,'G' ,'A'] 
    GTR = ef.mutation_matrix(pi_nuc, 4.49765,1,1,1,1,4.49765)

elif protein == '2ppn':
    #2ppn
    pi_nuc = [0.19246, 0.24559, 0.29365, 0.26830]
    GTR = ef.mutation_matrix(pi_nuc, 2.50275 ,1,1,1,1,2.50275)

elif protein == '1pek':
    #1pek
    pi_nuc = [0.20853, 0.34561, 0.25835, 0.18750]
    GTR = ef.mutation_matrix(pi_nuc, 0.90382 ,1,1,1,1, 0.90382)


#read in equilibrium sequence
with open("../data/eq_seqs/" + protein + "/" + protein + "_equilibrium_cdn_seq_t" + str(trial) + ".txt") as file:
    start_seq = file.readline().strip()
current_cdn_seq = mf.Codon_to_Cindex(start_seq)


###############################################################
#      Evolve based on thermodynamic  stability constraints            
###############################################################
rst_file = "../output/" + protein + "/"

full_ssF = np.zeros((num_subs, len(current_cdn_seq), 20))
full_ssdG = np.zeros((num_subs, len(current_cdn_seq), 20))

f = open(rst_file + protein + "_"+ NE +"_t" + str(trial) + '_main_rst_diff_eq_seq.txt', 'w') #create output file 
print("num_sub site old_codon new_codon old_aa new_aa fitness dG", flush = True, file = f)

for i in range(num_subs):
    #convert codon sequence to amino acid sequence 
    aa_seq = mf.Codon_to_AA(current_cdn_seq)
    
    #calculate site-specific fitness and dG vectors 
    all_sites_ssF, all_sites_ssdG = sf.full_ss_fitness(aa_seq, num_cores, ns_contact_map, alt_contact_maps)
    full_ssF[i] = all_sites_ssF
    full_ssdG[i] = all_sites_ssdG
    
    #save intial fitness and dG
    if i == 0:
        fitness = all_sites_ssF[0, aa_seq[0]]
        dG = all_sites_ssdG[0, aa_seq[0]]

        #sub site old_codon new_codon old_aa new_aa fitness dG
        print("-1" + " " + "--" + " " + "--" +  " " + "--" + " " + "--" + " " +  "--" + " " + 
              str(fitness) + " " + str(dG), flush = True, file = f)
    
    #calcualte transition probabilities
    Q =  ef.full_transition_vector(current_cdn_seq, all_sites_ssF, num_cores, Neff, GTR)
    
    #substitute
    site, new_codon = ef.next_substitution(Q)
    new_seq = np.array(current_cdn_seq)
    new_seq[site] = new_codon
    
    new_aa = mf.AminoAcid[mf.Codon_AA[mf.Codon[new_codon]]]
    fitness = all_sites_ssF[site, new_aa]
    dG = all_sites_ssdG[site, new_aa]
    
    #sub site old_codon new_codon old_aa new_aa fitness dG
    print(str(i) + " " + str(site) + " " + str(mf.Codon[current_cdn_seq[site]]) +  " " +
          str(mf.Codon[new_codon]) +" " + str(mf.Codon_AA[mf.Codon[current_cdn_seq[site]]]) + " " +  
          str(mf.Codon_AA[mf.Codon[new_codon]])  + " " + str(fitness) + " " +  str(dG), flush = True, file = f)
    
    current_cdn_seq = list(new_seq)

    if i % 50 == 0:
        np.save(rst_file + protein + "_"+ NE +"_t" + str(trial) + "_ssF_diff_eq_seq.npy", full_ssF)
        np.save(rst_file + protein + "_"+ NE +"_t" + str(trial) + "_ssdG_diff_eq_seq.npy", full_ssdG)

np.save(rst_file + protein + "_"+ NE +"_t" + str(trial) + "_ssF_diff_eq_seq.npy", full_ssF)
np.save(rst_file + protein + "_"+ NE +"_t" + str(trial) + "_ssdG_diff_eq_seq.npy", full_ssdG)

f.close()
