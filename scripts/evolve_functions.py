#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 13:59:49 2020

@author: nooryoussef

Functions required to evolve protein
"""
import numpy as np 
import misc_functions as mf 
from joblib import Parallel, delayed

def mutation_matrix(pi_nuc, a,b,c,d,e,f):
    '''
        Creates a mutation matrix. 
        
        pi_nuc = nucleotide frequencies [T,C,A,G]
        a: T <-> C   
        b: T <-> A 
        c: T <-> G 
        d: C <-> A 
        e: C <-> G 
        f: A <-> G 
    '''
    GTR = np.dot(np.matrix([ [0, a, b, c], 
                             [a, 0, d, e], 
                             [b, d, 0, f], 
                             [c, e, f, 0]]), np.diag(pi_nuc))
    #Fill in diagonal elements s.t. row sums to 0 
    for i in range(0, len(GTR)):
        GTR[i,i] = -np.sum(GTR[i,:])
    return GTR


def nuc_diff(source, target):
    '''
        Returns the nucleotide difference(s) between two codons.
        
        Returns a string of nucleotide difference between source and target codons
        source, target = three-letter sting represting a codons 
    '''
    return "".join( [source[i]+target[i] for i in range(len(source)) if source[i] != target[i]] )


def single_step_neighbour():
    '''
        makes a dictionary with codons as keys and thier single step neighbours as values
    '''
    single_step_neighbour = {}
    for codon in range(61):
        single_step_neighbour[codon] = []
        for codon2 in range(61):
            diff = nuc_diff(mf.Codon[codon], mf.Codon[codon2])
            if len(diff) == 2:
                single_step_neighbour[codon].append(codon2)
    return (single_step_neighbour)

single_step_neighbour = single_step_neighbour()

def site_specific_transition_vector(site, current_seq, ss_aaF, Neff, GTR):
    '''
        calculates the transition probabilities to all possible single-step mutations at a site given the current codon
        current_seq: codon index sequence
        ss_aaF: site specific amino acid fitness vector
        
        returns: a vector of the transition probabilities given the current codon at the site
    '''
    Q = np.zeros(61)
    current_codon = current_seq[site]
    current_aa = mf.AminoAcid[mf.Codon_AA[mf.Codon[current_codon]]]
    fit = ss_aaF[current_aa]

    #iterate over all single nucleotide mutations
    for new_codon in  single_step_neighbour[current_codon]:
        new_aa = mf.AminoAcid[mf.Codon_AA[mf.Codon[new_codon]]]
        
        diff = nuc_diff(mf.Codon[current_codon], mf.Codon[new_codon])
        n1 = mf.Nucleotide[diff[0]]; n2 = mf.Nucleotide[diff[1]]
        
        #synonymous mutation:
        if new_aa == current_aa:
            Q[new_codon] = GTR[n1,n2]
            
        #nonsynonymous mutation:
        else:
            new_fit = ss_aaF[new_aa]
            
            Sij = (new_fit - fit)/fit
            if abs(Sij) <= 1e-10:
                Q[new_codon] = GTR[n1,n2]
            else:
                Q[new_codon] = GTR[n1,n2]* 2 * Neff* ( (1 - np.exp(-2*Sij))/(1-np.exp(-4*Neff*Sij)))
    return (Q)


def full_transition_vector(current_seq, all_sites_aaF, num_cores, Neff, GTR):
    '''
        calculates the transition probabilities to all possible single-step mutations at all sites 
        current_seq: codon index sequence 
        ss_aaF: num sites x 20 vector of site-specific amino acid fitness values
    '''
    seq = np.array(current_seq)

    results = []
    results = Parallel(n_jobs=num_cores)(delayed(site_specific_transition_vector)(i, seq, all_sites_aaF[i], Neff, GTR) for i in range(len(seq)) )
    
    Q = np.array(results)
    return(Q)


def Q_matrix(F, GTR, Neff):
	'''
    creates a 61x61 instantaneous rate matrix based on the fitness vector (F)
    and the mutation model (GTR) and Neff 
    
    F: codon fitness vector [1x61]
	'''
	Q = np.zeros((61,61))
	for codon1 in range(0,61):
		for codon2 in range(0,61):
			diff = nuc_diff(mf.Codon[codon1], mf.Codon[codon2])
			if len(diff) == 2:
				Sij =  (F[codon2] - F[codon1])/F[codon1]
				n1 = mf.Nucleotide[diff[0]]
				n2 = mf.Nucleotide[diff[1]]
				if abs(Sij) <= 1e-10:
				    Q[codon1, codon2] = GTR[n1,n2]
				else:   
				    Q[codon1, codon2] = GTR[n1,n2] * 2 * Neff* ( (1 - np.exp(-2*Sij))/(1-np.exp(-4*Neff*Sij)))
	for i in range(0, 61):
		Q[i,i] = -np.sum(Q[i])
	return Q
	
    
def next_substitution(Q):
    '''
    Randomly draw substituted site and codon based on substitution probabilities 
    '''
    rate = np.sum(Q)
    TransitionProb = Q / rate
    X = np.random.multinomial(1, TransitionProb.reshape(len(TransitionProb)* 61)).reshape(len(TransitionProb), 61)
    site, codon = np.unravel_index(X.argmax(), X.shape)
    return (site, codon)

    