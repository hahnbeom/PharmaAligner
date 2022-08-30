import numpy as np
import torch,sys,os
import warnings
warnings.filterwarnings('ignore')

def read_mol2(mol2f):#get pharmacophore xyz
    sys.path.insert(0,'/home/hpark/programs/generic_potential')
    from Molecule import MoleculeClass
    from BasicClasses import OptionClass
    import Types

    option = OptionClass(['','-s',mol2f])
    molecule = MoleculeClass(mol2f,option)

    donorclass = [21,22,23,25,27,28,31,32,34,43]
    sp3donor = [21,22,34,43]
    acceptorclass = [22,26,33,34,36,37,38,39,40,41,42,43,47]
    aliphaticclass = [3,4] #3: CH2, 4: CH3; -> make [4] to be more strict (only CH3)

    names = [atm.name for atm in molecule.atms]
    xyz = molecule.xyz

    # get A,D,H
    lst = []
    for i,atm in enumerate(molecule.atms):
        xyz1 = xyz[i]

        if atm.aclass in donorclass and atm.has_H:
            lst.append(xyz[i])

        if atm.aclass in acceptorclass:
            lst.append(xyz[i])

        elif atm.aclass in aliphaticclass:
            lst.append(xyz[i])

    # get R as ring center (virtual atom)
    for i,ring in enumerate(molecule.rings_aro):
        xyz_ring = np.array([molecule.xyz[i] for i in ring.atms])
        ring_com = np.mean(xyz_ring,axis=0)

        xyz = np.concatenate([xyz,ring_com[None,:]],axis=0) # virtual position for the ring center
        lst.append(ring_com.tolist())
    lst = np.array(lst)

    return lst

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

def find_s(A,B):#get USR method of s
    s = 1/(1+(sum(abs(A-B)))/12)#larger s, higher the similariy 

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
        co_f12 = find_factor(x)#12 factors : mean,variance,skewness of ctd,cst,fct,ftf
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
                if i not in fail:
                    fail.append(i)#fail : similar confomers

    for i in range(all_conf_cnt):
        if i not in fail:
            result.append(i)

    result = np.array(result)
    return result

def cut_conformers(f):#input mol2 file has many conformers. get one confomer
    conf = []
    test_lst = []
    cnt = 0
    for l in open(f):
        if l.startswith('@<TRIPOS>MOLECULE'):
            cnt += 1
            if cnt > 1:
                tmp.close()
                conf.append(read_mol2('test_conf.mol2'))#get pharmacophore xyz
            tmp = open('test_conf.mol2','w')#test_conf.mol2 : one conformer
        tmp.write(l)

    return conf

f = sys.argv[1]#f : mol2 file
conf = cut_conformers(f)#conf : xyz list
result_index = usr_method(conf)#result_index : conformer numbers
make_conf_mol2(f,result_index)#get new mol2 file
print(result_index + 1)
