import numpy as np
def SPGR(T1, TR, flip):
    # flip in DEGREES
    E1 = np.exp(-float(TR) / T1)
    return ((1 - E1)*np.sin(np.radians(flip)))/(1-E1*np.cos(np.radians(flip)))

def SPGRe(T1, T2s, TR, TE, flip):
    return np.exp(-float(TE) / T2s) * SPGR(T1, TR, flip)

def IRSPGR(T1, TI, TR, N, flip):
    alpha = np.radians(flip)
    TS = TI+N*TR
    return (1. - 2.*np.exp(-float(TI)/T1) + np.exp(-float(TS)/T1)) * np.sin(alpha)
    
def MPRAGE(T1, TI, TD, TR, N, flip, Nk0 = 0):
    # Nk0 is the echo number for k0 line
    alpha = np.radians(flip)
    M0 = 1.
    TR = np.float64(TR)
    T1 = np.float64(T1)
    TI = np.float64(TI) - Nk0*TR
    TD = np.float64(TD)
    T1s = 1. / (1./T1 - np.log(np.cos(alpha))/TR)
    M0s = M0 * (1. - np.exp(-TR/T1)) / (1 - np.exp(-TR/T1s))
    
    A1 = M0s*(1 - np.exp(-(N*TR)/T1s))
    A2 = M0*(1 - np.exp(-TD/T1))
    A3 = M0*(1 - np.exp(-TI/T1))
    B1 = np.exp(-(N*TR)/T1s)
    B2 = np.exp(-TD/T1)
    B3 = -np.exp(-TI/T1)
    
    A = A3 + A2*B3 + A1*B2*B3
    B = B1*B2*B3   
    M1 = A / (1. - B)
    
    S1 = (M0s + (M1 - M0s)*np.exp(-(TR*Nk0)/(N*T1s)))*np.sin(alpha)
    return S1

def MP2RAGE(T1, TI1, TI2, TRSEG, TR, N, flip1, flip2, Nk0 = 0):
    
    a1 = np.radians(flip1)
    a2 = np.radians(flip2)
    M0 = 1
    TR = np.float64(TR)
    T1 = np.float64(T1)
    TI1 = np.float64(TI1)
    TI2 = np.float64(TI2)
    TRSEG = np.float64(TRSEG)
    
    R1 = 1/T1
    R1_1 = R1 - np.log(np.cos(a1))/TR
    R1_2 = R1 - np.log(np.cos(a2))/TR
    M0_1 = M0 * (1 - np.exp(-TR*R1))/(1 - np.exp(-TR*R1_1))
    M0_2 = M0 * (1 - np.exp(-TR*R1))/(1 - np.exp(-TR*R1_2))
    tau = N*TR
    delta_1 = TI1
    delta_2 = TI2 - TI1 - N*TR
    delta_3 = TRSEG - TI2 - N*TR
    
    A_1 = M0*(1 - np.exp(-delta_1 * R1))
    A_2 = M0*(1 - np.exp(-delta_2 * R1))
    A_3 = M0*(1 - np.exp(-delta_3 * R1))
    B_1 = np.exp(-delta_1 * R1)
    B_2 = np.exp(-delta_2 * R1)
    B_3 = np.exp(-delta_3 * R1)
    
    C_1 = M0_1*(1-np.exp(-tau * R1_1))
    C_2 = M0_2*(1-np.exp(-tau * R1_2))
    D_1 = np.exp(-tau * R1_1)
    D_2 = np.exp(-tau * R1_2)
    
    #Assume inversion efficiency is 1
    denom = 1 + B_1*D_1*B_2*D_2*B_3
    M1 = (A_1 - B_1*(A_3 + B_3*(C_2 + D_2*(A_2 + B_2*C_1))))/denom
    M2 = (A_2 + B_2*(C_1 + D_1*(A_1 - B_1*(A_3 + B_3*C_2))))/denom
    Mss =-(A_3 + B_3*(C_2 + D_2*(A_2 + B_2*(C_1 + D_1*A_1))))/denom
    return (M1, M2, Mss)
    
def SSFP(T1, T2, omega, TR, alpha, phi):
    # Omega - off-resonance in Hz
    # Alpha - flip-angle in degrees
    # Phi   - phase-cycling in degrees
    E1 = np.exp(-np.float64(TR) / T1)
    E2 = np.exp(-np.float64(TR) / T2)
    theta = 2.*np.pi*omega*TR+np.radians(phi)
    alphar = np.radians(alpha)
    
    G = GS(T1,T2,omega,TR,alpha)
    a = E2
    b = (E2*(1-E1)*(1+np.cos(alphar))) / (1-E1*np.cos(alphar)-E2**2*(E1-np.cos(alphar)))
    return  G * (1 - a*np.exp(-1j*theta)) / (1 - b*np.cos(theta))

def GS(T1, T2, omega, TR, alpha):
    E1 = np.exp(-np.float(TR) / T1)
    E2 = np.exp(-np.float(TR) / T2)
    theta = np.pi*omega*TR
    alphar = np.radians(alpha)
    return np.sqrt(E2)*np.exp(1.0j*(theta))*(1-E1)*np.sin(alphar)/(1.-E1*E2**2 - (E1 - E2**2)*np.cos(alphar))

def SSFP_MT(T1f, T2f, T1r, ):
    A = 1 + F - fw*E1r*(F+fk)
    sB = 1 + fk*(F-fw*E1r*(F+1))
    sC = F*(1-E1r)*(1-fk)

    n = sin(a)*(1-E2f*exp(-I*th))*((1-E1f)*B+C)
    d = A - B*E1f*cos(a) - E2f**2*(B*E1f-A*cos(a)) - E2f*cos(th)*(A-B*E1f)*(1+cos(a))
    mxy=nice_n/nice_d
    
    
def ernst(T1, TR):
    return np.degrees(np.arccos(np.exp(-TR/T1)))

def opt_despot1(T1, TR):   
    E1 = np.exp(-TR/T1)
    d1_lo = np.degrees(np.arccos((E1 + np.sqrt(2)*(1 - E1**2))/(2 - E1**2)))
    d1_hi = np.degrees(np.arccos((E1 - np.sqrt(2)*(1 - E1**2))/(2 - E1**2)))
    return d1_lo, d1_hi
    
def opt_despot2(T1, T2, TR):
    E1 = np.exp(-float(TR) / T1)
    E2 = np.exp(-float(TR) / T2)
    F = (1 - E1*E2)
    G = (E1 - E2)
    
    d2_lo = np.degrees(np.arccos((F*G+np.sqrt(2)*(F**2-G**2))/(2*F**2-G**2)))
    d2_hi = np.degrees(np.arccos((F*G-np.sqrt(2)*(F**2-G**2))/(2*F**2-G**2)))
    return d2_lo, d2_hi
    
def opt_despot2e(T1, T2, TR):
    E1 = np.exp(-float(TR) / T1)
    E2 = np.exp(-float(TR) / T2)
    F = (1 - E1*E2**2)
    G = (E1 - E2**2)
    
    d2_lo = np.degrees(np.arccos((F*G+np.sqrt(2)*(F**2-G**2))/(2*F**2-G**2)))
    d2_hi = np.degrees(np.arccos((F*G-np.sqrt(2)*(F**2-G**2))/(2*F**2-G**2)))
    return d2_lo, d2_hi

def sigma_T1(sigma, M0, T1, TR, alpha1, alpha2):
    a1 = np.radians(alpha1)
    a2 = np.radians(alpha2)
    s1 = np.sin(a1)
    s2 = np.sin(a2)
    c1 = np.cos(a1)
    c2 = np.cos(a2)
    E1 = np.exp(-TR / T1)
    s = np.abs(T1**2*sigma*np.sqrt((E1*c1 - 1)**2*s2**2 + (E1*c2 - 1)**2*s1**2)*(E1*c1 - 1)*(E1*c2 - 1)/
               (E1*M0*TR*(E1 - 1)*(c1 - c2)*(s1*s2)))
    return s

def sigma_T2(sigma, M0, T1, T2, TR, alpha1, alpha2):
    a1 = np.radians(alpha1)
    a2 = np.radians(alpha2)
    E1 = np.exp(-TR/T1)
    E2 = np.exp(-TR/T2)
    F  = (1 - E1*E2)
    G  = (E1 - E2)
    s1 = np.sin(a1)
    s2 = np.sin(a2)
    c1 = np.cos(a1)
    c2 = np.cos(a2)
    s = np.abs((T2**2*sigma*np.sqrt((F - G*c1)**2*s2**2 + (F - G*c2)**2*s1**2)*(F - G*c1)*(F - G*c2))/
               (E2*M0*TR*(1-E1)*(1-E1**2)*(c1 - c2)*s1*s2))
    return s

def sigma_T2e(sigma, M0, T1, T2, TR, alpha1, alpha2):
    a1 = np.radians(alpha1)
    a2 = np.radians(alpha2)
    E1 = np.exp(-TR/T1)
    E2 = np.exp(-TR/T2)
    F  = (1 - E1*E2**2)
    G  = (E1 - E2**2)
    s1 = np.sin(a1)
    s2 = np.sin(a2)
    c1 = np.cos(a1)
    c2 = np.cos(a2)
    s = np.abs((T2**2*sigma*np.sqrt((F - G*c1)**2*s2**2 + (F - G*c2)**2*s1**2)*(F - G*c1)*(F - G*c2))/
               (2*E2**2*M0*TR*(1-E1)*(1-E1**2)*(c1 - c2)*s1*s2))
    return s