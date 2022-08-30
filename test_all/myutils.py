import torch
import time

# Y:  B x N x 3
# Yp: N x 3
def Kabsch_batch(Y,Yp):

    B = Y.shape[0]
    comY = Y.mean(axis=1).unsqueeze(1)
    comYp = Yp.mean(axis=0)#[None,:].repeat(B,1) #B x 3

    Y = Y - comY
    Yp = Yp - comYp

    # Computation of the covariance matrix
    # put a little bit of noise to Y
    C = torch.einsum('bni,nj->bij',Y,Yp) #B x N x N
    
    # Computate optimal rotation matrix using SVD
    V, S, W = torch.svd(C) #B x N x N, B x N, B x N

    # get sign( det(V)*det(W) ) to ensure right-handedness
    d = torch.ones([B,3,3])
    d[:,:,-1] = torch.sign(torch.det(V) * torch.det(W))[:,None].repeat(1,3)

    # Rotation matrix U
    
    ## TODO: SLOW PART!  sometimes 0.0x ms -> 70 ms
    U = torch.einsum('bij,bkj->bik', d*V, W)
    #U = torch.bmm(d*V,W.transpose(1,2)) 
    
    rY = torch.einsum('bij,bjk->bik', Y, U)
    dY = torch.sum( torch.square(Yp-rY), axis=1) #Yp: N x 3; rY: B x N x 3

    rms = torch.sqrt( torch.sum(dY,dim=1) / Yp.shape[0] )

    #print("%.5f %.5f %.5f %.5f"%(t1-t0,t2-t1,t3-t2,t3-t0))
    #print("%.5f %.5f %.5f %.5f"%(t3-t2,t4-t3,t5-t4,t5-t2))

    return rms, U

def Kabsch(Y,Yp): # Yp: require_grads
    comY = Y.mean(axis=0)
    comYp = Yp.mean(axis=0)

    Y = Y - comY
    Yp = Yp - comYp

    # Computation of the covariance matrix
    # put a little bit of noise to Y
    C = torch.mm(Y.T, Yp)
    
    # Computate optimal rotation matrix using SVD
    V, S, W = torch.svd(C)

    # V: 3x3, det(V): 0-dim
    # get sign( det(V)*det(W) ) to ensure right-handedness
    d = torch.ones([3,3])
    d[:,-1] = torch.sign(torch.det(V) * torch.det(W))
    
    # Rotation matrix U
    U = torch.mm(d*V, W.T)

    rY = torch.einsum('ij,jk->ik', Y, U)
    dY = torch.sum( torch.square(Yp-rY), axis=1 )

    rms = torch.sqrt( torch.sum(dY) / Yp.shape[0] )

    return rms, U

def align(Y,U,com):
    if U.dim() == 3 and com.dim() == 2:
        B = U.shape[0]
        Z = Y.repeat(B,1,1) - com.unsqueeze(1)
        Z = torch.einsum( 'bij,bjk -> bik', Z, U) 
    elif U.dim() == 3 and com.dim() == 1:
        B = U.shape[0]
        Z = Y - com # B x N x 3
        Z = torch.einsum( 'bij,bjk -> bik', Z, U) 
    elif U.dim() == 2:
        Z = Y - com
        Z = torch.einsum( 'ij,jk -> ik', Z, U) 
    return Z

def make_pdb(atms,xyz,outf,header=""):
    out = open(outf,'w')
    if header != "":
        out.write(header)
        
    #ATOM      1  N   VAL A  33     -15.268  78.177  37.050  1.00 92.09      A    N
    form = "HETATM %5d %-3s UNK X %3d   %8.3f %8.3f %8.3f 1.00  0.00\n"
    for i,(atm,x) in enumerate(zip(atms,xyz)):
        #out.write("%-3s  %8.3f %8.3f %8.3f\n"%(atm,x[0],x[1],x[2]))
        out.write(form%(i,atm,1,x[0],x[1],x[2]))

    if header != "":
        out.write("ENDMDL\n")
    out.close()

def generate_pose(Y, keyidx, xyzfull, atms=[], epoch=0, report=False):
    make_pdb(atms,xyzfull,"init.pdb")
    Yp = xyzfull[keyidx]
    # find rotation matrix mapping Y to Yp

    T = torch.mean(Yp - Y, dim=0)

    com = torch.mean(Yp,dim=0)
    rms,U = rmsd(Y,Yp)
    
    Z = xyzfull - com
    T = torch.mean(Y - Yp, dim=0) + com
    
    Z = torch.einsum( 'ij,jk -> ik', Z, U.T) + T

    outf = "epoch%d.pdb"%epoch
    if report: make_pdb(atms,Z,outf,header="MODEL %d\n"%epoch)
    return Z
