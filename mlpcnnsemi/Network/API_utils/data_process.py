import numpy as np
import pandas as pd
import pepfeature as pep
import random
## Input: Aptamer Sequence 
## Output: vector with k-mer frequency
def count_kmers(read,k=3,k2=4):
    """Count kmer occurrences in a given read.

    Parameters
    ----------
    read : string
        A single DNA sequence.
    k : int
        The value of k for which to count kmers.

    Returns
    -------
    counts : dictionary, {'string': int}
        A dictionary of counts keyed by their individual kmers (strings
        of length k).

    Examples
    --------
    >>> count_kmers("GATGAT", 3)
    {'ATG': 1, 'GAT': 2, 'TGA': 1}
    """
    # Start with an empty dictionary
    for i in read:
        if i=='T':
            read=read.replace(i,'U')
    AUGC_LIST=['A','U','G','C']
    temp = random.randint(0,3)
    temp_AUGC  = AUGC_LIST[temp] 
    if read.find("N")!=-1:
        read = read.replace("N",temp_AUGC)
        # print(read)
    if read.find("B")!=-1:
        read = read.replace("B",temp_AUGC)
    counts={   'AAA':0,'AAU':0,'AAG':0,'AAC':0,
               'AUA':0,'AUU':0,'AUG':0,'AUC':0,
               'AGA':0,'AGU':0,'AGG':0,'AGC':0,
               'ACA':0,'ACU':0,'ACG':0,'ACC':0,
               'UAA':0,'UAU':0,'UAG':0,'UAC':0,
               'UUA':0,'UUU':0,'UUG':0,'UUC':0,
               'UGA':0,'UGU':0,'UGG':0,'UGC':0,
               'UCA':0,'UCU':0,'UCG':0,'UCC':0,
               'GAA':0,'GAU':0,'GAG':0,'GAC':0,
               'GUA':0,'GUU':0,'GUG':0,'GUC':0,
               'GGA':0,'GGU':0,'GGG':0,'GGC':0,
               'GCA':0,'GCU':0,'GCG':0,'GCC':0,
               'CAA':0,'CAU':0,'CAG':0,'CAC':0,
               'CUA':0,'CUU':0,'CUG':0,'CUC':0,
               'CGA':0,'CGU':0,'CGG':0,'CGC':0,
               'CCA':0,'CCU':0,'CCG':0,'CCC':0,
               
               'AAAA':0,'AAAU':0,'AAAG':0,'AAAC':0,
               'AAUA':0,'AAUU':0,'AAUG':0,'AAUC':0,
               'AAGA':0,'AAGU':0,'AAGG':0,'AAGC':0,
               'AACA':0,'AACU':0,'AACG':0,'AACC':0,
               'AUAA':0,'AUAU':0,'AUAG':0,'AUAC':0,
               'AUUA':0,'AUUU':0,'AUUG':0,'AUUC':0,
               'AUGA':0,'AUGU':0,'AUGG':0,'AUGC':0,
               'AUCA':0,'AUCU':0,'AUCG':0,'AUCC':0,
               'AGAA':0,'AGAU':0,'AGAG':0,'AGAC':0,
               'AGUA':0,'AGUU':0,'AGUG':0,'AGUC':0,
               'AGGA':0,'AGGU':0,'AGGG':0,'AGGC':0,
               'AGCA':0,'AGCU':0,'AGCG':0,'AGCC':0,
               'ACAA':0,'ACAU':0,'ACAG':0,'ACAC':0,
               'ACUA':0,'ACUU':0,'ACUG':0,'ACUC':0,
               'ACGA':0,'ACGU':0,'ACGG':0,'ACGC':0,
               'ACCA':0,'ACCU':0,'ACCG':0,'ACCC':0,
               
               'UAAA':0,'UAAU':0,'UAAG':0,'UAAC':0,
               'UAUA':0,'UAUU':0,'UAUG':0,'UAUC':0,
               'UAGA':0,'UAGU':0,'UAGG':0,'UAGC':0,
               'UACA':0,'UACU':0,'UACG':0,'UACC':0,
               'UUAA':0,'UUAU':0,'UUAG':0,'UUAC':0,
               'UUUA':0,'UUUU':0,'UUUG':0,'UUUC':0,
               'UUGA':0,'UUGU':0,'UUGG':0,'UUGC':0,
               'UUCA':0,'UUCU':0,'UUCG':0,'UUCC':0,
               'UGAA':0,'UGAU':0,'UGAG':0,'UGAC':0,
               'UGUA':0,'UGUU':0,'UGUG':0,'UGUC':0,
               'UGGA':0,'UGGU':0,'UGGG':0,'UGGC':0,
               'UGCA':0,'UGCU':0,'UGCG':0,'UGCC':0,
               'UCAA':0,'UCAU':0,'UCAG':0,'UCAC':0,
               'UCUA':0,'UCUU':0,'UCUG':0,'UCUC':0,
               'UCGA':0,'UCGU':0,'UCGG':0,'UCGC':0,
               'UCCA':0,'UCCU':0,'UCCG':0,'UCCC':0,
               
               'GAAA':0,'GAAU':0,'GAAG':0,'GAAC':0,
               'GAUA':0,'GAUU':0,'GAUG':0,'GAUC':0,
               'GAGA':0,'GAGU':0,'GAGG':0,'GAGC':0,
               'GACA':0,'GACU':0,'GACG':0,'GACC':0,
               'GUAA':0,'GUAU':0,'GUAG':0,'GUAC':0,
               'GUUA':0,'GUUU':0,'GUUG':0,'GUUC':0,
               'GUGA':0,'GUGU':0,'GUGG':0,'GUGC':0,
               'GUCA':0,'GUCU':0,'GUCG':0,'GUCC':0,
               'GGAA':0,'GGAU':0,'GGAG':0,'GGAC':0,
               'GGUA':0,'GGUU':0,'GGUG':0,'GGUC':0,
               'GGGA':0,'GGGU':0,'GGGG':0,'GGGC':0,
               'GGCA':0,'GGCU':0,'GGCG':0,'GGCC':0,
               'GCAA':0,'GCAU':0,'GCAG':0,'GCAC':0,
               'GCUA':0,'GCUU':0,'GCUG':0,'GCUC':0,
               'GCGA':0,'GCGU':0,'GCGG':0,'GCGC':0,
               'GCCA':0,'GCCU':0,'GCCG':0,'GCCC':0,
               
               'CAAA':0,'CAAU':0,'CAAG':0,'CAAC':0,
               'CAUA':0,'CAUU':0,'CAUG':0,'CAUC':0,
               'CAGA':0,'CAGU':0,'CAGG':0,'CAGC':0,
               'CACA':0,'CACU':0,'CACG':0,'CACC':0,
               'CUAA':0,'CUAU':0,'CUAG':0,'CUAC':0,
               'CUUA':0,'CUUU':0,'CUUG':0,'CUUC':0,
               'CUGA':0,'CUGU':0,'CUGG':0,'CUGC':0,
               'CUCA':0,'CUCU':0,'CUCG':0,'CUCC':0,
               'CGAA':0,'CGAU':0,'CGAG':0,'CGAC':0,
               'CGUA':0,'CGUU':0,'CGUG':0,'CGUC':0,
               'CGGA':0,'CGGU':0,'CGGG':0,'CGGC':0,
               'CGCA':0,'CGCU':0,'CGCG':0,'CGCC':0,
               'CCAA':0,'CCAU':0,'CCAG':0,'CCAC':0,
               'CCUA':0,'CCUU':0,'CCUG':0,'CCUC':0,
               'CCGA':0,'CCGU':0,'CCGG':0,'CCGC':0,
               'CCCA':0,'CCCU':0,'CCCG':0,'CCCC':0,
              }
    # Calculate how many kmers of length k there are
    num_kmers = len(read) - k + 1
    # Loop over the kmer start positions
    for i in range(num_kmers):
        # Slice the string to get the kmer
        kmer = read[i:i+k]
        # Add the kmer to the dictionary if it's not there
        # if kmer not in counts:
        #     # counts[kmer] = 0
        #     temp = random.randint(0,3)
        #     kmer[0] = AUGC_LIST[temp] if kmer[0] not in AUGC_LIST else kmer[0]
        #     kmer[1] = AUGC_LIST[temp] if kmer[1] not in AUGC_LIST else kmer[1]
        #     kmer[2] = AUGC_LIST[temp] if kmer[2] not in AUGC_LIST else kmer[2]
        #     kmer[3] = AUGC_LIST[temp] if kmer[3] not in AUGC_LIST else kmer[3]
            
        # Increment the count for this kmer
        counts[kmer] += 1/num_kmers
    
    if k2:
        num_kmers = len(read) - k2 + 1
        for i in range(num_kmers):
            # Slice the string to get the kmer
            kmer = read[i:i+k2]
            # Add the kmer to the dictionary if it's not there
            # if kmer not in counts:
            #     # counts[kmer] = 0
            #     temp = random.randint(0,3)
            #     kmer[0] = AUGC_LIST[temp] if kmer[0] not in AUGC_LIST else kmer[0]
            #     kmer[1] = AUGC_LIST[temp] if kmer[1] not in AUGC_LIST else kmer[1]
            #     kmer[2] = AUGC_LIST[temp] if kmer[2] not in AUGC_LIST else kmer[2]
            #     kmer[3] = AUGC_LIST[temp] if kmer[3] not in AUGC_LIST else kmer[3]
            # Increment the count for this kmer
            counts[kmer] += 1/num_kmers
        
        
    aptamer_list = []
    for val in counts.values():
        aptamer_list.append(val)

    return aptamer_list


## Input:  Protein sequence
## Output: Vector with protein feature
def protein_feature(seq):
    aa_20 = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']

    #count_amino_acids
    target=seq
    for i in target:
        if i=='X':
            target=target.replace(i,aa_20[random.randrange(20)])
    list_aa=[]
    num_A = target.count("A")
    list_aa.append(num_A)
    num_C = target.count("C")
    list_aa.append(num_C)
    num_D = target.count("D")
    list_aa.append(num_D)
    num_E = target.count("E")
    list_aa.append(num_E)
    num_F = target.count("F")
    list_aa.append(num_F)
    num_G = target.count("G")
    list_aa.append(num_G)
    num_H = target.count("H")
    list_aa.append(num_H)
    num_I = target.count("I")
    list_aa.append(num_I)
    num_K = target.count("K")
    list_aa.append(num_K)
    num_L = target.count("L")
    list_aa.append(num_L)
    num_M = target.count("M")
    list_aa.append(num_M)
    num_N = target.count("N")
    list_aa.append(num_N)
    num_P = target.count("P")
    list_aa.append(num_P)
    num_Q = target.count("Q")
    list_aa.append(num_Q)
    num_R = target.count("R")
    list_aa.append(num_R)
    num_S = target.count("S")
    list_aa.append(num_S)
    num_T = target.count("T")
    list_aa.append(num_T)
    num_V = target.count("V")
    list_aa.append(num_V)
    num_W = target.count("W")
    list_aa.append(num_W)
    num_Y = target.count("Y")
    list_aa.append(num_Y)
    print(list_aa)

    # The hydrophobicity values are from JACS, 1962, 84: 4240-4246. (C. Tanford).
    H01={'A':0.62,'C':0.29,'D':-0.90,'E':-0.74,'F':1.19,'G':0.48,'H':-0.40,'I':1.38,'K':-1.50,'L':1.06,'M':0.64,'N':-0.78,'P':0.12,'Q':-0.85,'R':-2.53,'S':-0.18,'T':-0.05,'V':1.08,'W':0.81,'Y':0.26}

    # Normalize (zero mean value; Eq. 4)
    avg_H01Val=0
    for i1 in H01.keys():
        avg_H01Val += H01[i1]/20
    sum_diff_H01Val=0
    for i2 in H01.keys():
        sum_diff_H01Val += (H01[i2] - avg_H01Val)**2
    sqrt_diff_H01Val=(sum_diff_H01Val/20)**0.5
    
    H1={}
    for i3 in H01.keys():
        H1[i3]=(H01[i3]-avg_H01Val)/sqrt_diff_H01Val
    # Check for "zero mean value"
    #H1_sum=0
    #for i in H1.values():
    #    H1_sum += i
    #print H1_sum/20
    
    # The hydrophilicity values are from PNAS, 1981, 78:3824-3828 (T.P.Hopp & K.R.Woods).
    H02={'A':-0.5,'C':-1.0,'D':3.0,'E':3.0,'F':-2.5,'G':0.0,'H':-0.5,'I':-1.8,'K':3.0,'L':-1.8,'M':-1.3,'N':0.2,'P':0.0,'Q':0.2,'R':3.0,'S':0.3,'T':-0.4,'V':-1.5,'W':-3.4,'Y':-2.3}

    # Normalize (zero mean value; Eq. 4)
    avg_H02Val=0
    for j1 in H02.keys():
        avg_H02Val += H02[j1]/20
    sum_diff_H02Val=0
    for j2 in H02.keys():
        sum_diff_H02Val += (H02[j2] - avg_H02Val)**2
    sqrt_diff_H02Val=(sum_diff_H02Val/20)**0.5
    
    H2={}
    for j3 in H02.keys():
        H2[j3]=(H02[j3]-avg_H02Val)/sqrt_diff_H02Val
    # Check for "zero mean value"
    #H2_sum=0
    #for i in H2.values():
    #    H2_sum += i
    #print H2_sum/20
    
    # The side-chain mass for each of the 20 amino acids.
    M0={'A':15.0,'C':47.0,'D':59.0,'E':73.0,'F':91.0,'G':1.0,'H':82.0,'I':57.0,'K':73.0,'L':57.0,'M':75.0,'N':58.0,'P':42.0,'Q':72.0,'R':101.0,'S':31.0,'T':45.0,'V':43.0,'W':130.0,'Y':107.0}

    # Normalize (zero mean value; Eq. 4)
    avg_M0Val=0
    for k1 in M0.keys():
        avg_M0Val += M0[k1]/20
    sum_diff_M0Val=0
    for k2 in M0.keys():
        sum_diff_M0Val += (M0[k2] - avg_M0Val)**2
    sqrt_diff_M0Val=(sum_diff_M0Val/20)**0.5
    
    M={}
    for k3 in M0.keys():
        M[k3]=(M0[k3]-avg_M0Val)/sqrt_diff_M0Val
    # Check for "zero mean value"
    #M_sum=0
    #for i in M.values():
    #    M_sum += i
    #print M_sum/20
    
    # The correlation function is given by the Eq. 3
    def theta_RiRj(Ri,Rj):
        return ((H1[Rj]-H1[Ri])**2+(H2[Rj]-H2[Ri])**2+(M[Rj]-M[Ri])**2)/3
    
    # Sequence order effect (Eq. 2)
    def sum_theta_val(seq_len,LVal,n):
        sum_theta_RiRj=0
        i=0
        while i < (seq_len-LVal):
            sum_theta_RiRj += theta_RiRj(target[i],target[i+n])
            #print i, seq[i], i+n, seq[i+n], theta_RiRj(seq[i],seq[i+n])
            i +=1
        return sum_theta_RiRj/(seq_len - n)

    LambdaVal=30
    if ((len(target)-LambdaVal) > 0):
        sum_all_aa_freq=0
        for aa in list_aa:
            #normalized occurrence frequency of the 20 amino acids
            sum_all_aa_freq += round(aa/len(target),3)
            
        num1=1
        all_theta_val=[]
        sum_all_theta_val=0
        while num1 < (int(LambdaVal)+1):
            tmpval=sum_theta_val(len(target),LambdaVal,num1)
            all_theta_val.append(tmpval)
            sum_all_theta_val += tmpval
            num1+=1
    

            # Denominator of the Eq. 6
        denominator_val=sum_all_aa_freq+(0.15*sum_all_theta_val)
            
        all_PseAAC1=[] # Eq. 5
            
        for val1 in list_aa:
            all_PseAAC1.append(round(((val1/20)/denominator_val),3))  #(1<= x <=20)  
        for val2 in all_theta_val:
            all_PseAAC1.append(round(((0.15*val2)/denominator_val),3))  #(21<= x <=20+landa)
    print(all_PseAAC1)
    print(len(all_PseAAC1))
    print("12345")
    

    return all_PseAAC1






#     # protein_list=[]
    
#     # df = pd.DataFrame([seq],columns=['Info_window_seq'])
#     # feat = pep.aa_all_feat.calc_df(dataframe=df, aa_column='Info_window_seq', Ncores=1, k=2)
#     # # print("111")
#     # # print(type(feat))
#     # # print(feat.values)
#     # a = feat.values.tolist()
#     # # print(a[0][1:])
#     # return a[0][1:]
    
# def protein_feature(seq):
#     protein_list=[]
    
#     df = pd.DataFrame([seq],columns=['Info_window_seq'])
#     feat = pep.aa_all_feat.calc_df(dataframe=df, aa_column='Info_window_seq', Ncores=1, k=2)
#     a = feat.values.tolist()
    
#     return a[0][1:]
     
def merge_apt_pro(test_list1,test_list2):
    res_list=np.concatenate((test_list1,test_list2))
    # res_list = [y for x in [test_list1, test_list2] for y in x]
    
    return res_list