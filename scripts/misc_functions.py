#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 13:11:37 2020

@author: nooryoussef

Useful functions
"""
import numpy as np 

AminoAcid = {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q':5,
             'E': 6, 'G': 7, 'H': 8, 'I': 9, 'L': 10, 'K': 11,
             'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19}

Codon = {0: 'CGT', 1: 'CGC', 2: 'CGA', 3: 'CGG', 4: 'AGA', 5: 'AGG',
         6: 'AAA', 7: 'AAG', 8: 'AAT', 9: 'AAC', 10: 'GAT', 11: 'GAC',
        12: 'GAA', 13: 'GAG', 14: 'CAA', 15: 'CAG', 16: 'CAT',
        17: 'CAC', 18: 'CCT', 19: 'CCC', 20: 'CCA', 21: 'CCG',
        22: 'TAT', 23: 'TAC', 24: 'TGG', 25: 'TCT', 26: 'TCC',
        27: 'TCA', 28: 'TCG', 29: 'AGT', 30: 'AGC', 31: 'ACT',
        32: 'ACC', 33: 'ACA', 34: 'ACG', 35: 'GGT', 36: 'GGC',
        37: 'GGA', 38: 'GGG', 39: 'GCT', 40: 'GCC', 41: 'GCA',
        42: 'GCG', 43: 'ATG', 44: 'TGT', 45: 'TGC', 46: 'TTT',
        47: 'TTC', 48: 'TTA', 49: 'TTG', 50: 'CTT', 51: 'CTC',
        52: 'CTA', 53: 'CTG', 54: 'GTT', 55: 'GTC', 56: 'GTA',
        57: 'GTG', 58: 'ATT', 59: 'ATC', 60: 'ATA'}

Nucleotide = {"T":0, "C":1, "G":2, "A":3}

Codon_AA =  {"TTT":"F", "TTC":"F", "TTA":"L", "TTG":"L",
             "TCT":"S", "TCC":"S", "TCA":"S", "TCG":"S",
             "TAT":"Y", "TAC":"Y", 
             "TGT":"C", "TGC":"C", "TGG":"W",
             "CTT":"L", "CTC":"L", "CTA":"L", "CTG":"L",
             "CCT":"P", "CCC":"P", "CCA":"P", "CCG":"P",
             "CAT":"H", "CAC":"H", "CAA":"Q", "CAG":"Q",
             "CGT":"R", "CGC":"R", "CGA":"R", "CGG":"R",
             "ATT":"I", "ATC":"I", "ATA":"I", "ATG":"M",
             "ACT":"T", "ACC":"T", "ACA":"T", "ACG":"T",
             "AAT":"N", "AAC":"N", "AAA":"K", "AAG":"K",
             "AGT":"S", "AGC":"S", "AGA":"R", "AGG":"R",
             "GTT":"V", "GTC":"V", "GTA":"V", "GTG":"V",
             "GCT":"A", "GCC":"A", "GCA":"A", "GCG":"A",
             "GAT":"D", "GAC":"D", "GAA":"E", "GAG":"E",
             "GGT":"G", "GGC":"G", "GGA":"G", "GGG":"G" }

### AA index to synonymous codon index ### 
aa_syn_cdns =  {0:  [39, 40, 41, 42],
                1:  [0, 1, 2, 3, 4, 5],
                2:  [8, 9],
                3:  [10, 11],
                4:  [44, 45],
                5:  [14, 15],
                6:  [12, 13],
                7:  [35, 36, 37, 38],
                8:  [16, 17],
                9:  [58, 59, 60],
                10: [48, 49, 50, 51, 52, 53],
                11: [6, 7],
                12: [43],
                13: [46, 47],
                14: [18, 19, 20, 21],
                15: [25, 26, 27, 28, 29, 30],
                16: [31, 32, 33, 34],
                17: [24],
                18: [22, 23],
                19: [54, 55, 56, 57]}

############# Useful Functions #############
def aa_index_to_aa(aa_idx_seq):
    '''
        Given an aa idx seqeucne, convert it to a aa sequence. 
    '''
    aa_seq = []
    for i in range(len(aa_idx_seq)):
        aa = aa_idx_seq[i]
        for idx in AminoAcid:
            if AminoAcid[idx] == aa:
                aa_seq.append(idx)
    return "".join(aa_seq)


def Codon_to_AA(seqCodon):
    '''
        Given a codon index sequence, convert it to an amino acid index sequence.
    '''
    return [ AminoAcid[Codon_AA[Codon[cdn]]] for cdn in seqCodon ]

def Codon_to_Cindex(CodonSeq):
    '''
        Given a codon sequence, convert it to a codon index sequence. 
    '''
    seqCindex = []
    for i in range(0,len(CodonSeq),3):
        cdn = CodonSeq[i:i+3]
        for idx in Codon:
            if Codon[idx] == cdn:
                seqCindex.append(idx)
    return seqCindex

def AminoAcid_to_AAindex(seq):
    '''
        Given an Amino acid sequence, convert it to an AA index sequence.
    '''
    seq_index = [ AminoAcid[i] for i in seq]
    return (seq_index)

def aa_to_cdn_seq(aa_seq):
    '''
    Given an amino acid sequence, returns a codon sequence 
    '''
    cdn_seq = []
    for aa in aa_seq:
        aa_idx = AminoAcid[aa]
        syn_cdns = aa_syn_cdns[AminoAcid[aa]]
        cdn_idx = syn_cdns[np.random.randint(0,len(syn_cdns))]
        cdn_seq.append(Codon[cdn_idx])
    return("".join(cdn_seq))

def convert_vector_aa_to_cdn(aa_vec):
    '''
    Given an amino acid site specific array, converts it to a codon site specific array 
    '''
    ss_cdn_vec = np.zeros(61)
    for aa in aa_syn_cdns:
        cdns = aa_syn_cdns[aa]
        ss_cdn_vec[cdns] = aa_vec[aa]
    return(ss_cdn_vec)        
    