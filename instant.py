import numpy as np
from numpy import dot
from scipy.linalg import solve, expm, block_diag
from . import bloch_base as bb

def SSFP(T1, T2, omega, TR, alphad, phi):
    alpha = np.atleast_1d(np.radians(alphad))
    omega = np.atleast_1d(omega)
    phi   = np.atleast_1d(phi)
    R = bb.relax(T1, T2)
    dims = (3, len(alpha), len(omega), len(phi))
    m_e = np.empty(dims)
    I = np.mat(np.identity(3))    
    m_inf = np.matrix([0., 0., 1]).T
    E = expm(-R*TR)
    E_e = expm(-R*TR/2.)
    RHS = (1 - np.exp(-TR/T1))*m_inf
    for o in range(dims[2]):
        do = 2.*np.pi*omega[o]
        O = bb.rotZ(do*TR)
        O_e = bb.rotZ(do*TR/2)
        for a in range(dims[1]):
            A = bb.rotY(-alpha[a])
            for p in range(dims[3]):
                P = bb.rotZ(np.radians(phi[p]))
                LHS = A - dot(P, dot(O, E))
                m_plus = solve(LHS, RHS)
                m_e[:,a,o,p] = np.squeeze(dot(O_e,dot(E_e,m_plus)))
    return np.squeeze(m_e)

def SSFP_analytic(T1, T2, psi, TR, alphad, phi):
    E1 = np.exp(-TR/T1)
    E2 = np.exp(-TR/T2)
    omega = np.radians(psi + phi)
    alpha = np.radians(alphad)
    n = np.sqrt(E2)*(E1 - 1)*(E2*np.exp(1j*omega) - 1)*np.exp(1j*psi/2)*np.sin(alpha)
    d = E1*E2**2 - E1*E2*np.cos(alpha)*np.cos(omega) - E1*E2*np.cos(omega) + E1*np.cos(alpha) - E2**2*np.cos(alpha) + E2*np.cos(alpha)*np.cos(omega) + E2*np.cos(omega) - 1
    return n/d
    
def SSFP_2c(T1_a, T2_a, T1_b, T2_b, tau_a, f_a, omega, flip, TR, spoil = True, TE = 0, phase = 0):    
    alpha = np.radians(flip)
    
    R = block_diag(bb.relax(T1_a, T2_a),
                   bb.relax(T1_b, T2_b))    
    K = bb.exchange(f_a, 1 - f_a, tau_a)
    if spoil:
        P = block_diag(bb.spoil(), bb.spoil())
    else:
        P = block_diag(bb.rotZ(np.radians(phase)), bb.rotZ(np.radians(phase)))
        TE = TR / 2
    
    I = np.mat(np.identity(6))    
    m_inf = np.mat([0., 0., f_a, 0., 0., 1 - f_a]).T
    m_ss   = np.empty((6, len(alpha), len(omega)))
    m_echo = np.empty((6, len(alpha), len(omega)))
    
    for o in range(len(omega)):
        do = omega[o]*2*np.pi
        O = block_diag(bb.inf_rotZ(do), bb.inf_rotZ(do))
        
        L = expm(-(R + O + K)*TR)
        L_e = expm(-(R + O + K)*TE)
        RHS1 = dot(P, dot(I - L, m_inf))
        for a in range(len(alpha)):
            A = block_diag(bb.rotY(alpha[a]), bb.rotY(alpha[a]))
            LHS = (I - dot(A,dot(P,L)))
            RHS2 = dot(A, RHS1)
            #print LHS
            #print RHS2.T
            ss = solve(LHS, RHS2)
            #echo = bb.bloch(L_e, ss, m_inf)
            #print 'SS', ss.T
            #print 'Echo', echo.T
            m_ss[:,a,o]   = np.squeeze(ss)
            #m_echo[:,a,o] = np.squeeze(echo)
    return bb.sum_mc(m_ss)

def SSFP_2c_corrected(T1_a, T2_a, T1_b, T2_b, tau_a, f_a, omega, flip, TR, spoil = True, TE = 0, phase = 0):    
    alpha = np.radians(flip)
    
    R = block_diag(bb.relax(T1_a, T2_a),
                   bb.relax(T1_b, T2_b))    
    K = bb.exchange(f_a, 1 - f_a, tau_a)
    if spoil:
        P = block_diag(bb.spoil(), bb.spoil())
    else:
        P = block_diag(bb.rotZ(np.radians(phase)), bb.rotZ(np.radians(phase)))
        TE = TR / 2
    
    I = np.mat(np.identity(6))    
    m_inf = np.mat([0., 0., f_a, 0., 0., 1 - f_a]).T
    m_ss   = np.empty((6, len(alpha), len(omega)))
    m_echo = np.empty((6, len(alpha), len(omega)))
    
    for o in range(len(omega)):
        do = omega[o]*2*np.pi
        O = block_diag(bb.inf_rotZ(do), bb.inf_rotZ(do))
        
        L = expm(-(R + O + K)*TR)
        L_e = expm(-(R + O + K)*TE)
        RHS = dot(I - L, m_inf)
        for a in range(len(alpha)):
            A = block_diag(bb.rotY(-alpha[a]), bb.rotY(-alpha[a]))
            LHS = (I - dot(L,dot(A,P)))
            #print LHS
            #print RHS2.T
            ss = solve(LHS, RHS)
            #echo = bb.bloch(L_e, ss, m_inf)
            #print 'SS', ss.T
            #print 'Echo', echo.T
            m_ss[:,a,o]   = np.squeeze(ss)
            #m_echo[:,a,o] = np.squeeze(echo)
    return bb.sum_mc(m_ss)


def SSFP_2c_analytic(T1_a, T2_a, T1_b, T2_b, tau_a, f_a, psi, phi, flip, TR):
    
    E_1a = np.exp(-TR/T1_a)
    E_1b = np.exp(-TR/T1_b)
    E_2a = np.exp(-TR/T2_a)
    E_2b = np.exp(-TR/T2_b)
    f_b = 1 - f_a
    k_ab = 1/tau_a
    Eab = np.exp(-TR*k_ab/(f_b))
    A = Eab - 1
    B = Eab*f_a+f_b
    C = A - B + 2
    
    omega = psi+phi
    co = np.cos(omega)
    so = np.sin(omega)
    m_ss = np.empty((6, len(flip)))
    for a in range(len(flip)):
        alpha = np.radians(flip[a])
        sa = np.sin(alpha)
        ca = np.cos(alpha)
        LHS = np.mat([[-C*E_2a *ca *co + 1, C*E_2a*so *ca, C*E_1a*sa, A*E_2b*f_a *ca *co, -A*E_2b*f_a*so *ca, -A*E_1b*f_a*sa],
                    [-C*E_2a*so, -C*E_2a *co + 1, 0, A*E_2b*f_a*so, A*E_2b*f_a *co, 0],
                    [-C*E_2a*sa *co, C*E_2a*sa*so, -C*E_1a *ca + 1, A*E_2b*f_a*sa *co, -A*E_2b*f_a*sa*so, A*E_1b*f_a *ca],
                    [A*E_2a*f_b *ca *co, -A*E_2a*f_b*so *ca, -A*E_1a*f_b*sa, -B*E_2b *ca *co + 1, B*E_2b*so *ca, B*E_1b*sa],
                    [A*E_2a*f_b*so, A*E_2a*f_b *co, 0, -B*E_2b*so, -B*E_2b *co + 1, 0],
                    [A*E_2a*f_b*sa *co, -A*E_2a*f_b*sa*so, A*E_1a*f_b *ca, -B*E_2b*sa *co, B*E_2b*sa*so, -B*E_1b *ca + 1]])
        RHS = np.mat([[f_a*(-A*E_1b*f_b + C*E_1a - 1)*sa], [0], [f_a*(A*E_1b*f_b - C*E_1a + 1) *ca], [f_b*(-A*E_1a*f_a + B*E_1b - 1)*sa], [0], [f_b*(A*E_1a*f_a - B*E_1b + 1) *ca]])
        #print LHS
        #print RHS.T
        mss = solve(LHS, RHS)
        m_ss[:,a]   = np.squeeze(mss)
    return bb.sum_mc(m_ss)

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