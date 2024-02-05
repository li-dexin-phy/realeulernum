import numpy as np
from scipy.linalg import expm

def composite_Wannier(ham, U_hat, V_hat, n_internal, n_occ, bandwidth, steps, step_size, test_steps):
    N_occ = ham.shape[0]*n_occ//n_internal
    eval, evec = np.linalg.eigh(ham)
    occvec = evec[:, :N_occ]*(1+0*1j)
    X_hat = np.diag(np.log(np.diag(U_hat)))*1j/2/np.pi
    Y_hat = np.diag(np.log(np.diag(V_hat)))*1j/2/np.pi
    uiter = occvec
    omega = np.zeros((N_occ, N_occ), dtype=complex)
    deltat = 0.001
    st = np.sin(deltat)
    ct = np.cos(deltat)
    emat = np.array([[ct, st], [-st, ct]])
    ematt = emat.T.conj()
    functional_G = np.zeros(steps)
    omegatest = np.zeros(test_steps)
    XXYY = X_hat @ X_hat + Y_hat @ Y_hat
    uxxyy = uiter.T.conj() @ XXYY @ uiter
    uxmat = uiter.T.conj() @ X_hat @ uiter
    uymat = uiter.T.conj() @ Y_hat @ uiter
    for i in range(steps):
        omegatotal = 0
        count = 0
        for j1 in range(N_occ):
            if j1+bandwidth < N_occ:
                if j1 < bandwidth-2:
                    end_bandwidth = (j1+1)*2
                else:
                    end_bandwidth = j1+bandwidth
            else:
                end_bandwidth = N_occ
            for j2 in range(j1+1, end_bandwidth):
                count += 1
                uuxxyy = uxxyy[:,[j1, j2]][[j1, j2],:]
                umatx = uxmat[:,[j1, j2]][[j1, j2],:]
                umaty = uymat[:,[j1, j2]][[j1, j2],:]
                tra = np.trace(ematt @ uuxxyy @ emat-uuxxyy)
                uuux = np.diag(umatx)
                uuuy = np.diag(umaty)
                vuux = np.vdot(uuux, uuux)
                vuuy = np.vdot(uuuy, uuuy)
                udx = np.diag(ematt @ umatx @ emat)
                udy = np.diag(ematt @ umaty @ emat)
                vdx = np.vdot(udx, udx)
                vdy = np.vdot(udy, udy)
                omega[j1, j2] = (tra-(vdx+vdy-vuux-vuuy))
                omegatotal += np.abs(omega[j1, j2])**2
        omegatotal = np.sqrt(omegatotal)
        for num in range(test_steps):
            du = omega/omegatotal*step_size*(num+1)
            du -= du.T.conj()
            uiter2 = uiter @ expm(du)
            uxxyy = uiter2.T.conj()@ XXYY @uiter2
            uxmat = uiter2.T.conj()@ X_hat @uiter2
            uymat = uiter2.T.conj()@ Y_hat @uiter2
            ux1 = np.diag(uxmat)
            uy1 = np.diag(uymat)
            vx = np.vdot(ux1, ux1)
            vy = np.vdot(uy1, uy1)
            omegatest[num] = np.real(np.trace(uxxyy)-vx-vy)
        num = np.argmin(omegatest, axis=0)
        du = omega/omegatotal*step_size*(num+1)
        du -= du.T.conj()
        uiter = uiter @ expm(du)
        uxxyy = uiter.T.conj() @ XXYY @ uiter
        uxmat = uiter.T.conj() @ X_hat @ uiter
        uymat = uiter.T.conj() @ Y_hat @ uiter
    return uiter
