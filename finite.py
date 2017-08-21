import numpy as np
from numpy import dot
from scipy.linalg import expm, solve, block_diag, inv, cho_factor, cho_solve
from numpy.linalg import norm, cond
from . import bloch_base as bb

def SSFP(T1, T2, omega, TR, Trf, alphad, phid):
    omega = np.atleast_1d(omega)
    alphad = np.atleast_1d(alphad)
    phid = np.atleast_1d(phid)
    alpha = np.radians(alphad)
    phi   = np.radians(phid)
    TE = (TR-Trf)/2.
    R = bb.relax(T1, T2)
    m_e = np.empty((len(alpha), len(omega), len(phi)), dtype=np.complex128)
    m_inf = np.mat([0., 0., 1.0]).T
    I = np.mat(np.identity(3))
    for o in range(len(omega)):
        do = 2.*np.pi * np.fmod(omega[o], 2./TR)
        O = bb.inf_rotZ(do)
        L = expm(-(R + O)*(TR-Trf))
        Le = expm(-(R + O)*TE)
        for a in range(len(alpha)):
            da = alpha[a] / Trf
            A = bb.inf_rotY(da)
            Ld = expm(-(R + O + A)*Trf)
            m_infd = solve((R + O + A), dot(R, m_inf))
            for p in range(len(phi)):
                P = bb.rotZ(phi[p])
                temp = solve((I - dot(Ld,dot(P,L))), (dot(Ld,dot(P,dot(I-L,m_inf))) + dot(I-Ld,m_infd)))
                echo = bb.bloch(Le, temp, m_inf)
                m_e[a,o,p] = np.squeeze(echo[0,0] + echo[1,0]*1j)   
    return np.squeeze(m_e)

def SSFP_2c(params, scan_params, ss=False, echo=True):
    omega = params['omega']
    TR = float(scan_params['TR'])
    Trf = float(scan_params['Trf'])
    alpha = scan_params['flip']
    
    R = block_diag(bb.relax(params['T1_a'], params['T2_a']),
                   bb.relax(params['T1_b'], params['T2_b']))    
    K = bb.exchange(float(params['f_a']), 1 - float(params['f_a']), float(params['tau_a']))
    if scan_params['spoil']:
        P = block_diag(bb.spoil(), bb.spoil())
        TE = float(scan_params['TE']) - Trf
    else:
        P = block_diag(bb.rotZ(scan_params['phase']), bb.rotZ(scan_params['phase']))
        TE = (TR-Trf)/2.
    
    I = np.mat(np.identity(6))    
    m_inf = float(params['PD']) * np.mat([0., 0., float(params['f_a']), 0., 0., 1 - float(params['f_a'])]).T
    m_ss   = np.empty((6, len(alpha), len(omega)))
    m_echo = np.empty((6, len(alpha), len(omega)))
    
    for o in range(len(omega)):
        do = omega[o]*2*np.pi
        
        O = block_diag(bb.inf_rotZ(do), bb.inf_rotZ(do))
        L = expm(-(R + O + K)*(TR-Trf))
        L_e = expm(-(R + O + K)*TE)
        for a in range(len(alpha)):
            da = alpha[a] / Trf
            A = block_diag(bb.inf_rotY(da), bb.inf_rotY(da))
            Ld = expm(-(R + O + A)*Trf)
            m_infd = solve((R + O + A), dot(R, m_inf))
            temp = solve((I - dot(Ld,dot(P,L))), (dot(Ld,dot(P,dot(I-L,m_inf))) + dot(I-Ld,m_infd)))
            m_ss[:,a,o] = np.squeeze(temp)
            temp2 = bb.bloch(L_e, temp, m_inf)
            m_echo[:,a,o] = np.squeeze(temp2)
    if ss and echo:
        return bb.sum_mc(m_ss), bb.sum_mc(m_echo)
    elif ss:
        return bb.sum_mc(m_ss)
    else:
        return bb.sum_mc(m_echo)
    
def SSFP_3c(p, scan_params, ss=False, echo=True):
    p_ab = {'T1_a':p['T1_a'], 'T2_a':p['T2_a'], 'T1_b':p['T1_b'], 'T2_b':p['T2_b'], 'tau_a':p['tau_a'], 'f_a':p['f_a']/(1. - p['f_c']), 'omega':p['omega'], 'PD':p['PD']*(1.-p['f_c'])}
    p_c  = {'T1':p['T1_c'], 'T2':p['T2_c'], 'omega':p['omega'], 'PD':p['PD'] * p['f_c']}
    m_ss_ab, m_e_ab = SSFP_2c(p_ab, scan_params, True, True)
    m_ss_c, m_e_c = SSFP(p_c, scan_params, True, True)
    if ss and echo:
        return bb.sum_mc(m_ss_ab + m_ss_c), bb.sum_mc(m_e_ab + m_e_c)
    elif ss:
        return bb.sum_mc(m_ss_ab + m_ss_c)
    else:
        return bb.sum_mc(m_e_ab + m_e_c)
    