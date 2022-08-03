import sys,os,glob
import numpy as np

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

    feats = {'xyz':xyz,'axes':axes,'wy':wy,'names':names,
             'As':As,'Ds':Ds,'Rs':Rs, 'Hs':Hs}
    #np.savez(mol2f[:-5]+'.npz', **feats)
    return feats

def run_a_tar(tarf):
    path = tarf.split('/')[-1][:-4]
    os.system('tar -xvf %s 1>/dev/null'%tarf)
    os.chdir(path)
    
    mol2s = glob.glob('*mol2')
    feats = {}
    for m in mol2s:
        words = m[:-5].split('_')
        prefix = '.'.join([words[k] for k in [1,-2,-1]])
        #if prefix == 'CA.0.918':
        feats[prefix] = read_ligand_mol2(m)
    os.chdir('..')

    outf = 'npzs/'+path.split('_')[1]+'.'+path.split('_')[-1]+'.npz'
    np.savez(outf,**feats)
    os.system('rm -rf %s'%path)

tarf = sys.argv[1]
run_a_tar(tarf)
