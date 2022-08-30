import sys,os
import numpy as np
import torch
from itertools import permutations
from myutils import Kabsch_batch, align, make_pdb
from optparse import OptionParser
import time
import multiprocessing as mp
torch.multiprocessing.set_sharing_strategy('file_system')

def init_from_parser():
    parser = OptionParser()
    parser.add_option("-m",help="npz file containing multiple entries",default=None,type='string')
    parser.add_option("-l",help="list of npz files",default=None,type='string')
    parser.add_option("-b",help="batch size",default=1,type='int') # this already use multiprocess...
    parser.add_option("-p",help="text file containing pharmacophore info",default=None,type='string')
    parser.add_option("--report_pdb",help="make aligned pdb",default=False,action='store_true')
    parser.add_option("-v",help="verbose log",default=0,type='int')
    parser.add_option("-n",help="report results for these",default=1,type='int')
    parser.add_option("-w",help="weights for nalign,Y (format [float,float] w/o space)",default='1.0,0.0',type='string')
    parser.add_option("--scorecut",help="score cut",default=1.0,type='float')
    parser.add_option("--np",help="number of multiprocess",default=1,type='int')
    parser.add_option("--debug",help="debug mode",default=False,action='store_true')
    parser.add_option("--outprefix",help="output prefix",default=None,type='string')
    
    if len(sys.argv) < 5:
        parser.print_help()
        
    option = parser.parse_args()[0]

    inputfs = []
    if option.m == None and option.l == None:
        parser.print_help()
        sys.exit('-m or -l not specified')

    if option.m != None:
        inputfs = [option.m]
    if option.l != None:
        inputfs += [l[:-1] for l in open(option.l)]
        
    if option.p == None:
        parser.print_help()
        sys.exit('-p not specified')

    option.weights = [float(a) for a in option.w.split(',')]
    return option, inputfs

def read_ligand(npz,key,debug=False):
    data = np.load(npz,allow_pickle=True)[key].item()

    return (torch.tensor(data['xyz']).float(),
            torch.tensor(data['axes']).float(),
            torch.tensor(data['wy']).float(),
            np.array(data['names']), data['As'], data['Ds'], data['Hs'], data['Rs'], data['dist'])

def cal_len(A,B):
    d = np.sqrt(np.sum(np.square(A-B)))
    if d == 0:
        return (0)
    elif d <= 4.8:
        return (1)
    elif d <= 7.2:
        return (2)
    elif d <= 9.5:
        return (3)
    elif d <= 11.9:
        return (4)
    elif d <= 14.3:
        return (5)
    else:
        return (6)

def compare_dist(dist,lst,xPH):#compare distance
    xPH = xPH.numpy()
    for i in range(len(lst) - 1):
        for j in range(i + 1, len(lst)):
            if dist[lst[i]][lst[j]] != cal_len(xPH[i],xPH[j]):
                return (0)
    return (1)

def find_searchlist(As,Ds,Rs,Hs,nPHcore,dist,xPH):
    searchlist = []

    As = list(permutations(As,nPHcore['A']))
    Ds = list(permutations(Ds,nPHcore['D']))
    Rs = list(permutations(Rs,nPHcore['R']))
    Hs = list(permutations(Hs,nPHcore['H']))
    
    #all_cnt : the number of available pharmacophore combination
    #search : after FuzCav(compare distance) method, the number of available pharmacophore combination
    all_cnt = 0
    search = 0
    for a in As:
        for d in Ds:
            for h in Hs:
                for r in Rs:
                    lst = list(a)+list(d)+list(h)+list(r)
                    all_cnt += 1
                    if compare_dist(dist,lst,xPH) == 1:
                         searchlist.append(lst)
    search = len(searchlist)
    return searchlist,all_cnt,search

def read_pharmacores(txtf):
    X = {'A':[],'D':[],'H':[],'R':[]}
    Y = {'A':[],'D':[],'H':[],'R':[]}
    n = 0
    for l in open(txtf):
        words = l[:-1].split()
        key = words[0]
        xyz = [float(word) for word in words[1:4]]
        axis = [float(word) for word in words[4:7]]
        X[key].append(xyz)
        Y[key].append(axis)
        n += 1
        
    ## make sure nPHcores are less than 6 and more than 3!!
    assert(n<6)
    assert(n>=3)

    X = {key:torch.tensor(X[key]) for key in X}
    Y = {key:torch.tensor(Y[key]) for key in Y}
    return X,Y

def calc_dy(y1,y2,types):
    y2 = y2/torch.norm(y2)
    dy = torch.sum(y1*y2,dim=2)

    for i,t in enumerate(types):
        if t == 2:
            dy[:,i] = 1.0 - abs(dy[:,i]) #0~1
        else:
            dy[:,i] = 0.5*(1.0-dy[:,i])
    return dy
    
def runner(inputf,key,opt):
    PHcoref = opt.p

    prefix = key#'.'.join(inputf.split('/')[-1].split('.')[:-1])
    
    t0 = time.time()
    xlig,ylig,wy,atms,As_l,Ds_l,Hs_l,Rs_l,dist = read_ligand(inputf, key, opt.debug)

    t1 = time.time()
    xPHcores, yPHcores = read_pharmacores(PHcoref) # text parsing

    nPHcore = {key:min(len(xPHcores[key]),len(idx)) for key,idx in zip(['A','D','H','R'],[As_l,Ds_l,Hs_l,Rs_l])}

    maxmatch = sum([nPHcore[key] for key in nPHcore])
    ntarget = sum([len(xPHcores[key]) for key in xPHcores])
    if maxmatch < ntarget:
        if opt.v > 3:
            print("not enough candidate pharmacophores for %s (target %d, matched-from-ligand %d)"%(prefix,ntarget,maxmatch))
        return [],-1,0,0

    # target
    xPHcores = torch.cat([xPHcores['A'],xPHcores['D'],xPHcores['H'],xPHcores['R']],axis=0)
    yPHcores = torch.cat([yPHcores['A'],yPHcores['D'],yPHcores['H'],yPHcores['R']],axis=0)

    # ordered as A->D->R->H
    searchlist,all_cnt,search = find_searchlist(As_l,Ds_l,Rs_l,Hs_l,nPHcore,dist,xPHcores)
    if opt.debug: print("found %d permutations"%len(searchlist))

    types = [0 for _ in range(nPHcore['A'])]+[1 for _ in range(nPHcore['D'])]+[2 for _ in range(nPHcore['H'])]+[3 for _ in range(nPHcore['R'])]

    sortable = []
    comPH = torch.mean(xPHcores,axis=0)

    nbatch = int(len(searchlist)/opt.b) + 1
    batchlist = [[a for i,a in enumerate(searchlist) if int(i/opt.b) == k] for k in range(nbatch)]
    while [] in batchlist:
        batchlist.remove([])

    for i,idx_s in enumerate(batchlist):
        xlig_b = torch.stack([xlig[idx] for idx in idx_s],axis=0)

        ylig_b = torch.stack([ylig[idx] for idx in idx_s],axis=0)
        wy_b   = torch.stack([wy[idx] for idx in idx_s],axis=0)
        
        t1a = time.time()
        rmsd,U = Kabsch_batch(xlig_b,xPHcores)
        t1b = time.time()
        
        com = torch.mean(xlig_b,axis=1)
        x_al = align(xlig,U,com) + comPH
        y_al = align(ylig_b,U,torch.zeros(3)) 

        ## prv way
        #dy = (yPHcores-y_al) # BxNx3
        ## not a fancy way but takes care of aromatics
        dy = calc_dy(yPHcores,y_al,types)

        Yscore = torch.mean(dy,axis=1)

        #print("%.5f %.5f %.5f"%(t1b-t1a,t1c-t1b,t1d-t1c)) # 0.06,0.29,0.11 ms per a single-batch
        if opt.v > 3:
            print(i,idx_s,rmsd,wy_b,dy)

        for args in zip(rmsd,Yscore,idx_s,torch.sum(wy_b,axis=-1).int(),x_al,y_al):
            score = args[0] + opt.weights[0]*args[1] + opt.weights[1]/(len(args[2])+0.001)
            sortable.append([score]+list(args))
            
    sortable.sort()
    sortable.reverse()

    outputs = []
    while True:
        (score,rmsd,Yscore,idx,ny,x_al,y_al) = sortable.pop()
        if score > opt.scorecut: break
        outputs.append((score,(Yscore,rmsd,1.0/len(idx)),prefix,atms,x_al))

        if opt.v > 2:
            t2 = time.time()
            form = " - %s: Rmsd/Nmatch/Yscore/naxis: %6.3f/%2d/%6.3f/%2d"
            l = form%(prefix,rmsd,len(idx),Yscore,ny)
            l += '; time spent (processing/alignment) %5.2f/%5.2f, ncomb=%d'%(t1-t0,t2-t1,len(searchlist))
            print(l)

    return outputs,0,all_cnt,search

def multirunner(args):
    inputf,key,opt = args

    try:
        outputs,err,all_cnt,search = runner(inputf,key,opt)
    except:
        print("failed %s %s"%(inputf,key))
        outputs,err,all_cnt,search = [],-2,0,0
        
    return outputs,err,all_cnt,search

def main(inputf,opt):
    t0 = time.time()
    outprefix = opt.outprefix
    if outprefix == None: outprefix = '.'.join(inputf.split('/')[-1].split('.')[:-1])

    if opt.debug:
        #key = list(np.load(inputf,allow_pickle=True))[0]
        key = 'CA.0.918'
        runner(inputf,key,opt)
        return
    else:
        ## multiprocessing is much faster than multibatch because of a weird slowdown in einsum notified above...
        launcher = mp.Pool(processes=opt.np)
        args = [(inputf,key,opt) for key in np.load(inputf,allow_pickle=True)]
        print("%s: launching %d molecules on %d processors"%(inputf,len(args),opt.np))
        outputs_l = launcher.map(multirunner,args)
#print(outputs_l)
    outputs = []
    nfail = 0
    nran = 0
    all_cnt = 0
    search = 0
    for a in outputs_l:
        outputs += a[0]
        nfail += (a[1]<0)
        nran += (a[1]==0)
        all_cnt += a[2]
        search += a[3]

    t1 = time.time()

    # select
    npick = min(opt.n,len(outputs))
    outputs.sort()

    print(f"Reporting {npick} models from total {len(outputs)} molecules;")
    for i,args in enumerate(outputs[:npick]):
        score,scores,prefix,atms,x_al = args
        print("%s.rank.%d: %8.3f (%8.3f/%8.3f/%8.3f) %s"%(outprefix,i,score,scores[0],scores[1],scores[2],prefix))
        if opt.report_pdb: make_pdb(atms,x_al,'%s.rank%d.al.pdb'%(outprefix,i))
    
    print("%s: Ran %d (%d of which failed) in %.2f secs"%(inputf,nran,nfail,t1-t0))
    print("all_cnt : %d" %all_cnt)#the number of available pharmacophore combination)
    print("search : %d" %search)#after FuzCav(compare distance) method, the number of available pharmacophore combination
    
if __name__ == "__main__":
    opt,inputfs = init_from_parser()

    print("Run on %d npzs..."%len(inputfs))
    for inputf in inputfs:
        print(inputf)
        main(inputf,opt)
