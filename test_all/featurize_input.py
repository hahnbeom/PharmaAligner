import sys,os,glob
import numpy as np
import warnings
warnings.filterwarnings('ignore')

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

def cal_dist(xyz):
    dist = []
    for i in range(len(xyz)):
        dist.append([])
        for j in range(len(xyz)):
            l = cal_len(xyz[i],xyz[j])
            dist[i].append(l)
    dist = np.array(dist)
    return (dist)

def read_ligand_mol2(mol2f):
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

    xyz = np.array(molecule.xyz)
    axes = np.zeros(xyz.shape)
    wy = np.zeros(xyz.shape[0])

    # get list of A & D
    As,Ds,Rs,Hs = [],[],[],[]
    for i,atm in enumerate(molecule.atms):
        xyz1 = xyz[i]
        
        if atm.aclass in donorclass and atm.has_H:
            Ds.append(i)
            for j,_ in atm.bonds:
                if molecule.atms[j].is_H:
                    xyz2 = xyz[j]
                    dv = xyz2-xyz1
                    axes[i] = dv/np.sqrt(np.sum(dv*dv))
                    wy[i] = float(atm.aclass not in sp3donor)
            
        if atm.aclass in acceptorclass:
            As.append(i)
            axis = np.zeros(3)
            for j,_ in atm.bonds:
                xyz2 = xyz[j]
                axis += xyz1-xyz2
            axes[i] = axis/np.sqrt(np.sum(axis*axis))
            wy[i] = float(atm.aclass not in sp3donor)

        elif atm.aclass in aliphaticclass:
            Hs.append(i)
            axis = np.zeros(3)
            for j,_ in atm.bonds:
                if not molecule.atms[j].is_H:
                    xyz2 = xyz[j]
                    axis += xyz1-xyz2
            axes[i] = axis/np.sqrt(np.sum(axis*axis))
            wy[i] = 1.0
            
    # get R as ring center (virtual atom)
    for i,ring in enumerate(molecule.rings_aro):
        xyz_ring = np.array([molecule.xyz[i] for i in ring.atms])
        ring_com = np.mean(xyz_ring,axis=0)
        ring_axis = np.cross(ring_com-xyz_ring[0],ring_com-xyz_ring[1])
        
        Rs.append(len(axes)) #before append
        wy = np.concatenate([wy,np.array([1.0])],axis=0)
        xyz = np.concatenate([xyz,ring_com[None,:]],axis=0) # virtual position for the ring center
        axes = np.concatenate([axes,ring_axis[None,:]],axis=0) # virtual position for the ring center
        names.append('Ring%d'%i)

    #get dist
    dist = cal_dist(xyz)

    feats = {'xyz':xyz,'axes':axes,'wy':wy,'names':names,
             'As':As,'Ds':Ds,'Rs':Rs, 'Hs':Hs, 'dist':dist}
    #np.savez(mol2f[:-5]+'.npz', **feats)
    return feats

def run_mol2(mol2,names):#get npz
    conf = []
    test_lst = []
    cnt = 0
    feats = {}
    for l in open(mol2):
        if l.startswith('@<TRIPOS>MOLECULE'):
            cnt += 1
            if cnt > 1:
                tmp.close()
                feats[names[cnt - 2]] = read_ligand_mol2('test_conf.mol2')
            tmp = open('test_conf.mol2','w')
        tmp.write(l)
    path = mol2.split('/')[-1]
    path1 = 'npzs/' + path.split('.')[0] + '/'
    outf = path1 + path.replace('mol2','npz')
    np.savez(outf,**feats)

#os.system('rm -rf %s'%path)

mol2 = sys.argv[1]#mol2 file
f = open(sys.argv[2])#name_list.txt : all names of conformers
names = f.read().splitlines()
f.close()
run_mol2(mol2,names)#get npz
