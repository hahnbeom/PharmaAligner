import sys
from rdkit import Chem
from rdkit.Chem import AllChem

def main(pdb,n):

    m1 = Chem.MolFromPDBFile(pdb)
    m2 = Chem.MolFromPDBFile(pdb)

    m1 = Chem.AddHs(m1)
    m2 = Chem.AddHs(m2)

    Chem.SanitizeMol(m2)

    cids = AllChem.EmbedMultipleConfs(m2, numConfs=n+1)
    AllChem.AlignMolConformers(m2)

    pos = m1.GetConformer().GetPositions()

    conf0 = m2.GetConformer(0)

    nadd = m2.AddConformer(conf0, True) # attach 0-th to the end #assignID=True <- add to the end
    
    conf  = m2.GetConformer(0) # replace 0-th; call by reference
    for i,x in enumerate(pos):
        conf.SetAtomPosition(i,x)

    Chem.MolToPDBFile(m2,"conformers.pdb")
    
if __name__ == "__main__":
    pdb = sys.argv[1]
    n = int(sys.argv[2])

    # should interface through PDB because of stupid rdkit sanitization...
    main(pdb,n)
