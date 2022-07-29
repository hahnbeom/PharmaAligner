import sys
import numpy as np

def pdb2crd(pdbfile,opt,as_numpy=False,include_chain=False,resno_as_int=True,read_het=False):
    pdbcont=open(pdbfile)
    crd={}
    for line in pdbcont:
        
        if not (line[:4] =='ATOM' or (line.startswith('HETATM') and read_het)):
            continue
        if resno_as_int:
            resno = int(line[22:26])
            if include_chain:
                resno = '%s.%d'%(line[21],resno)
        else:
            resno = line[22:26].strip()
            if include_chain:
                resno = '%s.%s'%(line[21],resno)
            
        restype = line[16:20].strip()
        atmtype = line[12:16].strip()
        if opt == 'CA':
            if line[12:16] == ' CA ':
                v = [float(line[30+i*8:38+i*8]) for i in range(3)]
                if as_numpy: v = np.array(v)
                crd[resno] = v
        elif opt == 'CB':
            if (restype == 'GLY' and line[12:16] == ' CA ') or line[12:16] == ' CB ':
                v = [float(line[30+i*8:38+i*8]) for i in range(3)]
                if as_numpy: v = np.array(v)
                crd[resno] = v
        else:
            if resno not in crd:
                crd[resno] = {}
            v = [float(line[30+i*8:38+i*8]) for i in range(3)]
            if as_numpy: v = np.array(v)
            crd[resno][atmtype] = v
    pdbcont.close()
    return crd

def main(pdb,f):
    crd = pdb2crd(pdb,opt='all',include_chain=True,read_het=True)

    for l in open(f):
        if l.startswith('#'): continue
        words = l[:-1].split()
        t = words[0]
        tip = words[1].split('-')
        bases = [word.split('-') for word in words[2:]]

        y = np.zeros(3)
        if t == 'R':
            ring = [tip]+bases
            xring = np.array([crd[a[0]][a[1]] for a in ring])
            x = np.mean(xring,axis=0)
            y = np.cross(x-xring[0],x-xring[1])
        
        else:
            x = crd[tip[0]][tip[1]]
            for b in bases:
                y += crd[b[0]][b[1]]
            y = x - y
            
        y /= np.sqrt(np.dot(y,y,))
    
        print(t,' %8.3f'*3%tuple(x)+' %8.3f'*3%tuple(y)+' #%s'%words[1])
    
if __name__ == "__main__":
    pdb = sys.argv[1]
    f = sys.argv[2]
    main(pdb, f)
