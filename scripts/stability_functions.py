#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 12:23:51 2020

@author: nooryoussef
Functions used in stability calculations
"""
import numpy as np 
import numba as nb
import  misc_functions as mf
from joblib import Parallel, delayed


#import scipy.constants as sc
#kt = (sc.R/4184)*(20+273.15)
kt = 0.6

#Contact potentials from Miyazawa and Jernigan 1996
MJ85 = np.array(
[[-0.13,  0.43,  0.28,  0.12,  0.00,  0.08,  0.26, -0.07,  0.34, -0.22, -0.01,  0.14,  0.25,  0.03,  0.10, -0.06, -0.09, -0.09,  0.09, -0.10,  0.00],
 [ 0.43,  0.11, -0.14, -0.72,  0.24, -0.52, -0.74, -0.04, -0.12,  0.42,  0.35,  0.75,  0.31,  0.41, -0.38,  0.17, -0.35, -0.16, -0.25,  0.30,  0.00],
 [ 0.28, -0.14, -0.53, -0.30,  0.13, -0.25, -0.32, -0.14, -0.24,  0.53,  0.30, -0.33,  0.08,  0.18, -0.18, -0.14, -0.11,  0.06, -0.20,  0.50,  0.00],
 [ 0.12, -0.72, -0.30,  0.04,  0.03, -0.17, -0.15, -0.22, -0.39,  0.59,  0.67, -0.76,  0.65,  0.39,  0.04, -0.31, -0.29,  0.24,  0.00,  0.58,  0.00],
 [ 0.00,  0.24,  0.13,  0.03, -1.06,  0.05,  0.69, -0.08, -0.19,  0.16, -0.08,  0.71,  0.19, -0.23,  0.00, -0.02,  0.19,  0.08,  0.04,  0.06,  0.00],
 [ 0.08, -0.52, -0.25, -0.17,  0.05,  0.29, -0.17, -0.06, -0.02,  0.36,  0.26, -0.38,  0.46,  0.49, -0.42, -0.14, -0.14,  0.08, -0.20,  0.24,  0.00],
 [ 0.26, -0.74, -0.32, -0.15,  0.69, -0.17, -0.03,  0.25, -0.45,  0.35,  0.43, -0.97,  0.44,  0.27, -0.10, -0.26,  0.00,  0.29, -0.10,  0.34,  0.00],
 [-0.07, -0.04, -0.14, -0.22, -0.08, -0.06,  0.25, -0.38,  0.20,  0.25,  0.23,  0.11,  0.19,  0.38, -0.11, -0.16, -0.26,  0.18,  0.14,  0.16,  0.00],
 [ 0.34, -0.12, -0.24, -0.39, -0.19, -0.02, -0.45,  0.20, -0.29,  0.49,  0.16,  0.22,  0.99, -0.16, -0.21, -0.05, -0.19, -0.12, -0.34,  0.19,  0.00],
 [-0.22,  0.42,  0.53,  0.59,  0.16,  0.36,  0.35,  0.25,  0.49, -0.22, -0.41,  0.36, -0.28, -0.19,  0.25,  0.21,  0.14,  0.02,  0.11, -0.25,  0.00],
 [-0.01,  0.35,  0.30,  0.67, -0.08,  0.26,  0.43,  0.23,  0.16, -0.41, -0.27,  0.19, -0.20, -0.30,  0.42,  0.25,  0.20, -0.09,  0.24, -0.29,  0.00],
 [ 0.14,  0.75, -0.33, -0.76,  0.71, -0.38, -0.97,  0.11,  0.22,  0.36,  0.19,  0.25,  0.00,  0.44,  0.11, -0.13, -0.09,  0.22, -0.21,  0.44,  0.00],
 [ 0.25,  0.31,  0.08,  0.65,  0.19,  0.46,  0.44,  0.19,  0.99, -0.28, -0.20,  0.00,  0.04, -0.42, -0.34,  0.14,  0.19, -0.67, -0.13, -0.14,  0.00],
 [ 0.03,  0.41,  0.18,  0.39, -0.23,  0.49,  0.27,  0.38, -0.16, -0.19, -0.30,  0.44, -0.42, -0.44,  0.20,  0.29,  0.31, -0.16,  0.00, -0.22,  0.00],
 [ 0.10, -0.38, -0.18,  0.04,  0.00, -0.42, -0.10, -0.11, -0.21,  0.25,  0.42,  0.11, -0.34,  0.20,  0.26,  0.01, -0.07, -0.28, -0.33,  0.09,  0.00],
 [-0.06,  0.17, -0.14, -0.31, -0.02, -0.14, -0.26, -0.16, -0.05,  0.21,  0.25, -0.13,  0.14,  0.29,  0.01, -0.20, -0.08,  0.34,  0.09,  0.18,  0.00],
 [-0.09, -0.35, -0.11, -0.29,  0.19, -0.14,  0.00, -0.26, -0.19,  0.14,  0.20, -0.09,  0.19,  0.31, -0.07, -0.08,  0.03,  0.22,  0.13,  0.25,  0.00],
 [-0.09, -0.16,  0.06,  0.24,  0.08,  0.08,  0.29,  0.18, -0.12,  0.02, -0.09,  0.22, -0.67, -0.16, -0.28,  0.34,  0.22, -0.12, -0.04, -0.07,  0.00],
 [ 0.09, -0.25, -0.20,  0.00,  0.04, -0.20, -0.10,  0.14, -0.34,  0.11,  0.24, -0.21, -0.13,  0.00, -0.33,  0.09,  0.13, -0.04, -0.06,  0.02,  0.00],
 [-0.10,  0.30,  0.50,  0.58,  0.06,  0.24,  0.34,  0.16,  0.19, -0.25, -0.29,  0.44, -0.14, -0.22,  0.09,  0.18,  0.25, -0.07,  0.02, -0.29,  0.00],
 [ 0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00]])

############# Energy Calculations #############
@nb.jit(nopython = True)
def getG(aa_seq, contact_map):
    '''
        Calculates the free energy of a sequence (aaSeq) given a structure (ContactMap).
        Free energy calculated as the sum of potentials between amino acids in contact. 
        
        contact_map: specifies residues in contact [type = array]
        aa_seq: amino acid index sequence [type = array]
    '''
    G = 0 
    for site1 in range(len(aa_seq)):
        for site2 in range(site1, len(aa_seq)):
            if contact_map[site1,site2] == True:
                G +=  MJ85[aa_seq[site1], aa_seq[site2]]
    return G

def getGalt(aa_seq, alt_maps):
    '''
        Calculates the free energy of a given sequence (aaSeq) in the alternative structures (alt_maps).
        
        Returns G_alt: list of free energies in each alt structure (ordered).
    '''
    Galt = [getG(aa_seq, alt_maps) for alt_maps in alt_maps.values()]
    return Galt


def Fitness(seq, is_cdn, ns_contact_map, alt_maps):
    '''
        sequence: index sequence. Codon or amino acid index [type = array]
        is_cdn: specify if index sequence is codon or AA [1 => codon, 0 => AA]
    '''    
    # if sequence is in codon idx, convert to amino acid idx sequence
    if is_cdn:
        seq = mf.Codon_to_AA(seq)
        
    Gns = getG(seq, ns_contact_map) #G in Native structure
    Galt =  getGalt(seq, alt_maps) #G in alt structures
    
    dG = Gns - np.mean(Galt) + (np.var(Galt) / (2*kt)) +(kt*np.log(3.4**len(seq))) #delta G
    fitness = np.exp(-1.*dG/kt)/(1+ np.exp(-1.*dG/kt)) #Prob of folding = fitness
    
    return(fitness, Gns, Galt, dG)

@nb.jit(nopython = True)
def simple_getG(site, G, new_aaSeq, old_aaSeq, ns_contact_map):
    '''
        A faster way of calculating G if it has already been caluclated for a sequence that differs at a single site
        
        site: the site where new_aaSeq and old_aaSeq differ
        G: free energy of oldSeq in ContactMap
        new_aaSeq, old_aaSeq: Amino acid idx sequence 
    '''
    Gnew = G
    for site2 in range(len(old_aaSeq)):
        if ns_contact_map[site,site2] == True:
            # subtract old interactions and add new ones 
            Gnew +=  MJ85[new_aaSeq[site],new_aaSeq[site2]]- MJ85[old_aaSeq[site],old_aaSeq[site2]] 
    return(Gnew)

def simple_getGalt(site, G, new_aaSeq, old_aaSeq, alt_maps):
    '''
        site: the site where newSeq and oldSeq differ
        G: free energy of oldSeq in ContactMap
        new_aaSeq, old_aaSeq: Amino acid idx sequence 
    '''
    Galt = [simple_getG(site, G[i], new_aaSeq, old_aaSeq, contact_map) for i, contact_map in enumerate(alt_maps.values())]
    return Galt

def simple_Fitness(site, Gns, Galt, new_seq, old_seq, ns_contact_map, alt_maps, is_cdn):
    '''
        new_seq, , old_seq: codon idx sequences
        site: the site where newSeq and oldSeq differ
        G: free energy of oldSeq in ContactMap
        Galt: list of free energies in Alt Structures
    '''
    if is_cdn:
        new_seq = np.array(mf.Codon_to_AA(new_seq))
        old_seq = np.array(mf.Codon_to_AA(old_seq))
    
    Gns = simple_getG(site, Gns, new_seq, old_seq, ns_contact_map) #G in Native structure
    Galt =  simple_getGalt(site, Galt, new_seq, old_seq, alt_maps) #G in alt structures
    
    dG = Gns - np.mean(Galt) + (np.var(Galt) / (2*kt)) +(kt*np.log(3.4**len(new_seq))) #delta G
    fitness = np.exp(-1.*dG/kt)/(1+ np.exp(-1.*dG/kt)) #Prob of folding = fitness
    return(fitness, dG)

def ss_fitness(current_seq, site,  ns_contact_map, alt_maps):
        '''
        calculates the site specific amino acid fitness and dG 
        current_seq : an amino acid idx sequence 
        '''
        current_aa = current_seq[site]
        ssF = np.zeros(20)
        ssdG = np.zeros(20)
        
        Pold, GNSold, GALTold, dGold = Fitness(current_seq, 0, ns_contact_map, alt_maps)
        ssF[current_aa] = Pold
        ssdG[current_aa] = dGold
        
        for aa in [x for x in range(20) if x != current_aa]:
            new_seq = np.array(current_seq); new_seq[site] = aa
            F, dG = simple_Fitness(site, GNSold, GALTold, new_seq,  current_seq, ns_contact_map, alt_maps, 0)
            ssF[aa] = F
            ssdG[aa] = dG
        return((ssF, ssdG))

def full_ss_fitness(current_seq, num_cores, ns_contact_map, alt_maps):
        '''
        calculates the site specific amino acid fitness and dG 
        current_seq : an amino acid idx sequence 
        '''
        seq = np.array(current_seq)
        full_ssF = np.zeros((len(seq), 20))
        full_ssdG = np.zeros((len(seq), 20))
        
        results = []
        results = Parallel(n_jobs=num_cores)(delayed(ss_fitness)(seq, i,  ns_contact_map, alt_maps) for i in range(len(current_seq)) )
        
        for Idx, i in enumerate(results):
            full_ssF[Idx] = i[0]
            full_ssdG[Idx] = i[1]
            
        return(full_ssF, full_ssdG)

