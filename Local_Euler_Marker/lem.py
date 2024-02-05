import numpy as np
import pf

def rspace_euler(ham, U_hat, V_hat, unitary, n_internal, n_occ):
    N_coordinate = ham.shape[0]//n_internal
    N_occ = N_coordinate*n_occ
    xindex = np.mat(np.log(np.diag(U_hat)[::n_internal]))
    yindex = np.mat(np.log(np.diag(V_hat)[::n_internal]))
    eval, evec = np.linalg.eigh(ham)
    occvec = evec[:, :N_occ]*(1+0*1j)
    proj = occvec @ occvec.T.conj()
    projpx = U_hat.T.conj() @ proj @ U_hat
    projpy = V_hat.T.conj() @ proj @ V_hat
    pxpyp_Local = proj @ (projpx @ projpy-projpy @ projpx)
    k_x = np.mat(np.arange(N_coordinate)%np.sqrt(N_coordinate)).T
    k_y = np.mat(np.arange(N_coordinate)//np.sqrt(N_coordinate)).T
    ukR = np.exp(k_x@xindex+k_y@yindex)/np.sqrt(N_coordinate)
    ukRn = np.zeros((N_occ, N_occ), dtype = complex)
    ukRn[::n_occ,::n_occ] = ukR
    ukRn[1::n_occ,1::n_occ] = ukR
    pxpyp_Wannier = ukRn.T.conj() @ unitary.T.conj() @ pxpyp_Local @ unitary @ ukRn # wannier basis
    local = np.zeros(N_coordinate)
    for i in range(N_coordinate):
        local[i] = (pf.pfaffian(pxpyp_Wannier[i*n_occ:(i+1)*n_occ, i*n_occ:(i+1)*n_occ])).real/2/np.pi*N_coordinate
    return local
