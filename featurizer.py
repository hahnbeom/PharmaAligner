import os,sys
import numpy as np
import scipy
import scipy.spatial
import multiprocessing as mp

def grab_features_from_rdkit(pdb,mol2f):
    from rdkit import Chem
    from rdkit.Chem import ChemicalFeatures
    from rdkit import RDConfig
    
    fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    
    m = Chem.MolFromMol2File(pdb)
        
    feats = factory.GetFeaturesForMol(m)
    natm = len(m.GetAtoms())
    
    funcs = ['Acceptor','Donor','Aromatic']
    types = {}
    for i,f in enumerate(feats):
        fam = f.GetFamily()
        if fam not in funcs: continue

        ifunc = funcs.index(fam)
        for idx in f.GetAtomIds():
            if idx not in types: types[idx] = []
            types[idx].append(ifunc)

    functypes = np.zeros((natm,len(funcs)))
    names = []
    elems = []

    bonds = m.GetBonds()
    edges = [[b.GetBeginAtomIdx(),b.GetEndAtomIdx()] for b in bonds]
        
    xyz = m.GetConformer().GetPositions() #numpy array
     
    for i,atm in enumerate(m.GetAtoms()):
        ri = atm.GetPDBResidueInfo()
        names.append(ri.GetName().strip())
        elems.append(atm.GetSymbol())
        if i in types:
            functypes[i][types[i]] = 1

    feats = {'names':names, 'functypes':functypes, 'xyz':xyz, 'elems':elems, 'edges':edges}

    feats = aromatic_correction(feats, mol2)

    return feats

def aromatic_correction(feats, mol2):
    anames = feats['names']
    functypes = feats['functypes']

    # add aromatic property
    read_cont = False
    for l in open(mol2):
        if l.startswith('@<TRIPOS>ATOM'):
            read_cont = True
            continue
        if l.startswith('@<TRIPOS>BOND'):
            break
        if not read_cont: continue
        aname = l.split()[1]
        atype = l.split()[5]
        if aname in anames and '.ar' in atype:
            feats['functypes'][anames.index(aname)][2] = 1
            print('add aromaticity to %s'%(aname))
            
    return feats

def grab_features_from_mol2gen(mol2f):
    sys.path.insert(0,'/home/hpark/programs/generic_potential')
    from Molecule import MoleculeClass
    from BasicClasses import OptionClass
    import Types
    #from AtomTypeClassifier import FunctionalGroupClassifier

    option = OptionClass(['','-s',mol2f])
    molecule = MoleculeClass(mol2f,option)
    #classifier = FunctionalGroupClassifier()
    #classifier.apply_to_molecule(molecule)
    #molecule.report_functional_grps(sys.stdout)

    donorclass = [21,22,23,25,27,28,31,32,34,43]
    acceptorclass = [22,26,33,34,36,37,38,39,40,41,42,43,47]

    names = [atm.name for atm in molecule.atms]

    xyz = np.array(molecule.xyz)
    elems = [Types.ATYPES_REG[atm.atype] for atm in molecule.atms]
    functypes = np.zeros((len(names),3))
    edges = np.array([[bond.atm1,bond.atm2] for bond in molecule.bonds])
    axes= np.zeros((len(names),3)) # default is zero
    
    for i,atm in enumerate(molecule.atms):
        is_donor = False
        is_acceptor = False
        axis = []
        
        xyz1 = xyz[i]
        
        if atm.aclass in donorclass and atm.has_H:
            is_donor = True
            for j,_ in atm.bonds:
                if molecule.atms[j].is_H:
                    xyz2 = xyz[j]
                    axis = xyz2-xyz1
            
        if atm.aclass in acceptorclass:
            is_acceptor = True

            if len(axis) == 0:
                axis = np.zeros(3)
                for j,_ in atm.bonds:
                    xyz2 = xyz[j]
                    axis += xyz1-xyz2

        #if is_donor or is_acceptor:
        #    print(atm.name, is_donor, is_acceptor, axis)
            
        # append only if axis is defined correctly
        if len(axis) > 0:
            if is_acceptor: functypes[i][0] = 1
            if is_donor: functypes[i][1] = 1
            axes[i] = axis

    for ring in molecule.rings_aro:
        xyz_ring = np.array([molecule.xyz[i] for i in ring.atms])
        ring_com = np.mean(xyz_ring)
        ring_axis = np.cross(ring_com-xyz_ring[0],ring_com-xyz_ring[1])
        
        for i in ring.atms:
            functypes[i][2] = 1
            axes[i] = ring_axis
        
    feats = {'names':names, 'functypes':functypes, 'xyz':xyz, 'elems':elems, 'edges':edges, 'axes':axes}
    return feats

def find_AAfunctype(atmname):
    acceptors = ['ASP.OD1','ASP.OD2','GLU.OE1','GLU.OE2',
                 'HIS.ND1','HIS.NE2',
                 'SER.OG','THR.OG1','TYR.OH','ASN.OD1','GLN.OE1']
    
    donors = ['LYS.NZ','ARG.NH1','ARG.NH2',
              'HIS.ND1','HIS.NE2','TRP.NE1',
              'SER.OG','THR.OG1','TYR.OH','ASN.ND2','GLN.NE2']
    
    aros = ['PHE.CG','PHE.CD1','PHE.CD2','PHE.CE1','PHE.CE2','PHE.CZ',
            'TYR.CG','TYR.CD1','TYR.CD2','TYR.CE1','TYR.CE2','TYR.CZ',
            'TRP.CG','TRP.CD1','TRP.CD2','TRP.NE1','TRP.CE2','TRP.CE3','TRP.CZ2','TRP.CZ3','TRP.CH2']

    
    if atmname.split('.')[-1] in ['O','OXT'] or atmname in acceptors: 
        return 0 # acceptor
    elif (atmname.split('.')[-1] in ['N'] and atmname != 'PRO.N') or atmname in donors: 
        return 1 # donor
    elif atmname in aros:
        return 2
    return -1

def compatible(type1, type2s):
    if type1 == 0 and type2s[1] == 1:
        return True
    if type1 == 1 and type2s[0] == 1:
        return True
    elif type1 == 2 and type2s[2] == 1:
        return True
    else:
        return False

def get_pharmacophores_from_complex(holopdb,lig_functypes,ligname='LG1'):
    xyz_lig = []
    xyz_rec = []
    resnos = []
    rec_functypes = []
    atmname_lig = []
    atmname_rec = []
    
    for l in open(holopdb):
        if not (l.startswith('ATOM') or l.startswith('HETATM')): continue
        resname = l[16:20].strip()
        resno = l[21]+'.'+l[22:26].strip()
        atmname = resname + '.' + l[12:16].strip()
        
        xyz = [float(l[30:38]),float(l[38:46]),float(l[46:54])]
        
        if resname == ligname:
            xyz_lig.append(xyz)
            atmname_lig.append(l[12:16].strip())
        else:
            xyz_rec.append(xyz)
            resnos.append(resno)
            rec_functypes.append(find_AAfunctype(atmname))
            atmname_rec.append((resno,atmname))
            
    xyz_lig = np.array(xyz_lig)
    xyz_rec = np.array(xyz_rec)

    kd_lig = scipy.spatial.cKDTree(xyz_lig)
    kd_rec = scipy.spatial.cKDTree(xyz_rec)
    indices = kd_rec.query_ball_tree(kd_lig, 4.0)

    contacts = []
    for i,js in enumerate(indices):
        contacts += [(i,j) for j in js] 
    
    # pick closest contact per-ligatom as representative
    #by_ligatom = {}
    legit_ligatoms = []
    for i,j in contacts:
        if j >= len(lig_functypes) or (sum(lig_functypes[j]) == 0): continue
        
        dv = xyz_rec[i] - xyz_lig[j]
        d = np.sqrt(np.dot(dv,dv))
        #print(j,d,atmname_lig[j],rec_functypes[i],atmname_rec[i])
        if not compatible(rec_functypes[i],lig_functypes[j]): continue 
        
        #if j not in by_ligatom: by_ligatom[j] = []
        if j not in legit_ligatoms: legit_ligatoms.append(j)
        #by_ligatom[j].append((d,i,dv))
        
    # only those having direct contacts
    return legit_ligatoms

def report(feats,outpdb):
    legit = feats['legit']
    #out = open(outpdb,'w')
    #ATOM      1  N   PRO A   1     -71.473 -31.540  77.865  1.00  0.72           N
    #HETATM  42  H19 ELS X  42      12.217  22.505  63.129  1.00  0.00
    form = 'HETATM %4d  %-3s%4s X %3d    %8.3f%8.3f%8.3f  1.00  0.00'
    ctr = -1
    for name,t,xyz,axis in zip(feats['names'],feats['functypes'],feats['xyz'],feats['axes']):
        ctr += 1
        if ctr in legit:
            print(form%(ctr+1,name,'LEG',ctr+1,xyz[0],xyz[1],xyz[2]))
        elif sum(t) > 0:
            print(form%(ctr+1,name,'NLG',ctr+1,xyz[0],xyz[1],xyz[2]))
        else:
            print(form%(ctr+1,name,'ELS',ctr+1,xyz[0],xyz[1],xyz[2]))
        

def run(trg):
    os.chdir(trg)

    try:
        mol2 = '%s_ligand.mol2'%trg
        pdb = '%s.holo.pdb'%trg
        #feats = grab_features_from_rdkit(pdb,mol2)
        feats = grab_features_from_mol2gen(mol2)

        feats['legit'] = get_pharmacophores_from_complex(pdb, feats['functypes'])
        
        #report(feats,'test.pdb')
        outnpz = mol2[:4]+'.phcore.npz'
        np.savez(outnpz, **feats)
        
    except:
        print("failed %s"%trg)
        pass
    os.chdir('..')
            
trgs = [l[:-1] for l in open(sys.argv[1])]
a = mp.Pool(processes=20)
a.map(run,trgs)
