import numpy as np
import torch
import sys
import os

def read_mol2(mol2):#get conformer xyz without atom 'H'
    read_cont = False
    xyz = []
    for l in open(mol2):
        if l.startswith('@<TRIPOS>ATOM'):
            read_cont = True
            xyz.append([])
            continue
        if l.startswith('@<TRIPOS>BOND'):
            read_cont = False
            continue

        if read_cont:
            words = l[:-1].split()
            if words[1].startswith('H'):#without atom 'H'
                continue
            xyz[-1].append([float(word) for word in words[2:5]])

    for i,x in enumerate(xyz):
        xyz[i] = np.array(xyz[i])

    return xyz

def cal_distance(X,Y):
    dx = torch.unsqueeze(X,0) - torch.unsqueeze(Y,0)
    d = torch.sqrt(torch.sum(dx**2, 2))
    return d

def get(xyz,factor):#get mean,variance,skewness
    d = cal_distance(xyz,factor).tolist()
    mean = np.mean(d)
    variance = np.var(d)
    skew = (np.mean((d-mean)**3))/(variance**(3.0/2.0))
    return mean,variance,skew

def find_factor(xyz):#find ctd,cst,fct,ftf/return mean,variance,skewness
    lst = []
    factor4 = []

    ctd = torch.tensor(np.mean(xyz,axis=0))#ctd
    X = torch.tensor(xyz)
    d_ctd = cal_distance(X,ctd)

    cst = torch.tensor(xyz[torch.argmin(d_ctd,dim=1)])#cst
    fct = torch.tensor(xyz[torch.argmax(d_ctd,dim=1)])#fct
    d_fct = cal_distance(X,fct)

    ftf = torch.tensor(xyz[torch.argmax(d_fct,dim=1)])#ftf

    lst.extend(get(X,ctd))
    lst.extend(get(X,cst))
    lst.extend(get(X,fct))
    lst.extend(get(X,ftf))
    lst = np.array(lst)
#factor4 = np.array([ctd.tolist(),cst.tolist(),fct.tolist(),ftf.tolist()])
    return lst

def find_s(A,B):
    s = 1/(1+(sum(abs(A-B)))/12)

    return s

def make_conf_mol2(f,lst):#make new conformers
    cnt = -1
    new_conf = open('new_conformers.mol2','w')
    for l in open(f):
        if l.startswith('@<TRIPOS>MOLECULE'):
            cnt += 1
        if cnt in lst:
            new_conf.write(l)
    new_conf.close()

def usr_method(conf):#factor : mean,variance,skewness
    fail = []
    result= []
    factors = []
    similarity  = []
    for x in conf:
        co_f12 = find_factor(x)#12 factors : mean,varianc,skewness of ctd,cst,fct,ftf
        factors.append(co_f12)
    factors = np.array(factors)

    for j,y in enumerate(factors):
        line = []
        for i,x in enumerate(factors):
            if j == i:
                line.append(1)
                continue
            line.append(find_s(y,x))
        similarity.append(line)

    all_conf_cnt = len(similarity)
    for j,y in enumerate(similarity):
        for i in range(j + 1, len(similarity)):
            if y[i] > 0.6:
                if i not in fail:#fail : similar conformers
                    fail.append(i)

    for i in range(all_conf_cnt):
        if i not in fail:
            result.append(i)

    result = np.array(result)
    return result

conf = read_mol2(sys.argv[1])#conf : xyz list
result_index = usr_method(conf)#result_index : conformer numbers
make_conf_mol2(sys.argv[1],result_index)#get new mol2 file
print(result_index + 1)
