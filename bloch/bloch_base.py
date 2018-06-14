import numpy as np

def rotX(alpha):
    return np.matrix([[1., 0., 0.],[0., np.cos(alpha), np.sin(alpha)],[0., -np.sin(alpha), np.cos(alpha)]])

def rotY(alpha):
    return np.matrix([[np.cos(alpha), 0., -np.sin(alpha)],[0., 1., 0.],[np.sin(alpha), 0., np.cos(alpha)]])

def rotZ(phi):
    return np.mat([[np.cos(phi), -np.sin(phi), 0.],[np.sin(phi), np.cos(phi), 0.],[0., 0., 1.]])

def inf_rotX(dalpha):
    return np.mat([[0., 0., 0.],[0., 0., dalpha],[0., -dalpha, 0.]])

def inf_rotY(dalpha):
    return np.mat([[0., 0., -dalpha],[0., 0., 0.],[dalpha, 0., 0.]])

def inf_rotZ(do):
    return np.mat([[0, do, 0],[-do, 0, 0], [0, 0, 0]])

def spoil():
    return np.mat([[0., 0., 0.],[0., 0., 0.],[0., 0., 1.]])

def relax(T1, T2):
    return np.mat(np.diagflat([1/T2, 1/T2, 1/T1]))

def bloch(L, m_0, m_inf):
    return np.dot(L,(m_0 - m_inf)) + m_inf

def exchange(f_a, f_b, tau_a):
    if ((f_a == 0) or (f_b == 0)): #Actually have 1 component, so no exchange
        return np.zeros((6, 6))
    else:
        tau_b = f_b * tau_a / f_a
        k_ab = 1/tau_a
        k_ba = 1/tau_b
        K = np.mat(np.diagflat([k_ab, k_ab, k_ab, k_ba, k_ba, k_ba]) -
                   np.diagflat([k_ba, k_ba, k_ba], 3) -
                   np.diagflat([k_ab, k_ab, k_ab], -3))
        return K

def sum_mc(M):
    m_tot = M[0:3, ...].copy()
    for i in range(3, M.shape[0], 3):
        m_tot += M[i:i+3, ...]
    return m_tot

def sig_mag(M):
    s = np.sqrt(np.sum(M[0:2,...]**2, axis=0))
    return s

def sig_norm(M):
    s = sig_mag(M)
    s /= np.mean(s, axis=0)
    return s

def sig_phase(M):
    p = np.arctan2(M[1,...], M[0,...])
    return p
    