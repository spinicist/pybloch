import numpy as np
from numpy import dot
from scipy.linalg import expm, solve
from scipy.optimize import nnls
import despot

def echo_train(TE, ETL):
    return np.linspace(TE,TE*ETL,ETL)

def relax1c(params, scan_params):
    T1 = params['T1']
    T2 = params['T2']
    TE = scan_params['TE']
    ETL = scan_params['ETL']
    
    L = np.mat(np.diagflat([np.exp(-TE/T2), np.exp(-TE/T2), np.exp(-TE/T1)]))
    
    m_n = np.mat([1., 0., 0.]).T
    m_inf = np.mat([0., 0., 1.]).T
    
    m = np.empty((3, ETL))
    for i in range(ETL):
        m_n = dot(L, m_n - m_inf) + m_inf
        m[:, i] = np.squeeze(m_n)
    return m

def bases(T1, T2_low, T2_high, nT2, sp):
    T2 = np.logspace(np.log10(T2_low), np.log10(T2_high), nT2)
    b = np.empty((nT2, sp['ETL']))
    for i in range(nT2):
        p = {'T1': T1, 'T2': T2[i]}
        b[i, :] = despot.sig_mag((relax1c(p, sp)))
    return b, T2

def relax3c(params, scan_params):
    T1_a = params['T1_a']
    T1_b = params['T1_b']
    T1_c = params['T1_c']
    T2_a = params['T2_a']
    T2_b = params['T2_b']
    T2_c = params['T2_c']
    f_a  = params['f_a']
    f_c  = params['f_c']
    f_b  = 1. - f_a - f_c
    tau_a = params['tau_a']
    tau_b = f_b * tau_a / f_a
    k_ab = 0
    k_ba = 0
    if ((tau_a > 0) and (f_a > 0) and (f_b > 0)):
        k_ab = 1/tau_a;
        k_ba = 1/tau_b;
    
    
    PD = 1.
    TE = scan_params['TE']
    ETL = scan_params['ETL']
    
    D = np.mat(np.diagflat([1/T2_a, 1/T2_a, 1/T1_a, 1/T2_b, 1/T2_b, 1/T1_b, 1/T2_c, 1/T2_c, 1/T1_c]) +
               np.diagflat([k_ab, k_ab, k_ab, k_ba, k_ba, k_ba, 0, 0, 0]) -
               np.diagflat([k_ba, k_ba, k_ba, 0, 0, 0], 3) - np.diagflat([k_ab, k_ab, k_ab, 0, 0, 0], -3))
    L = np.mat(expm(-D * TE))
    m_n = np.mat([f_a, 0., 0., f_b, 0., 0., f_c, 0., 0.]).T
    m_inf = np.mat([0., 0., f_a, 0., 0., f_b, 0., 0., f_c]).T
    
    m = np.empty((9, ETL))
    for i in range(ETL):
        m_n = dot(L, m_n - m_inf) + m_inf
        m[:, i] = np.squeeze(m_n)
    
    m2 = m[0:3,:] + m[3:6,:] + m[6:9,:]
    return m