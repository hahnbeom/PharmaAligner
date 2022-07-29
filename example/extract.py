import sys
import numpy as np
sys.path.insert(0,'/home/hpark/util3')
from pdbutils import pdb2crd

crd = pdb2crd(sys.argv[1],opt='all',include_chain=True,read_het=True)
f = sys.argv[2]

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
    
