import numpy as np
import math as m
import matplotlib.pyplot as plt
from scipy.stats import norm


# Black-Scholes Formula for a European Call Option
#--------------------------------------------------------
def black_scholes_put(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put = -S0 * norm.cdf(-d1) + K * np.exp(-r * T) * norm.cdf(-d2)
    return put


#BINOMIAL MODELS
#----------------------------------------------------------


# CRR Binomial Model for an American Option
#-----------------------------------------------
def crr(S0, K, r, sigma, T, N):

    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    # Building the stock tree
    ST_list=[S0]
    for j in range(1,N+1,1):
        box =[]
        for i in range(j+1):
         box.append(S0 * (u**i) * (d**(j-i)))
    
        ST_list.append(box)
    

    # option value at maturity 
    ST = np.array(ST_list[-1])
    C = np.maximum(-ST+K, 0.0)

    # backward induction
    discount = np.exp(-r * dt)
    for i in range(N, 1, -1):
        # Il nuovo array C avrà 'i' elementi (riduzione di 1 elemento)
        C_nuovo = np.zeros(i)
        
        # Ciclo interno per calcolare ogni nodo nel tempo precedente
        # 'j' va da 0 a i-1 (indice del nodo al tempo precedente)
        for j in range(i):
            # C[j+1] è il valore in caso di UP-move dal nodo j
            # C[j] è il valore in caso di DOWN-move dal nodo j
            valore_atteso = p * C[j+1] + (1-p) * C[j]
            C_nuovo[j] = np.maximum(discount * valore_atteso,max(-ST_list[i-1][j]+K,0))
    
        
        # Sostituisci il vecchio array C con il nuovo array calcolato
        C = C_nuovo
    
    valore_atteso = p * C[1] + (1-p) * C[0]
    C = discount * valore_atteso

    
    # Dopo N iterazioni, C conterrà un solo elemento: il prezzo all'inizio (t=0)
    return C




#Tian Model for American Option
#---------------------------------------------
def tian(S0, K, r, sigma, T, N):
    dt = T / N
    M=np.exp(r*dt)
    V=np.exp(sigma**2*dt)
    u=0.5*(M*V)*(V+1+np.sqrt(V**2+2*V-3))
    d=0.5*(M*V)*(V+1-np.sqrt(V**2+2*V-3))
    p = (np.exp(r * dt) - d) / (u - d)

  # Building the stock tree
    ST_list=[S0]
    for j in range(1,N+1,1):
        box =[]
        for i in range(j+1):
         box.append(S0 * (u**i) * (d**(j-i)))
    
        ST_list.append(box)
    

    # option value at maturity (CALL)
    ST = np.array(ST_list[-1])
    C = np.maximum(-ST+K, 0.0)

    # backward induction
    discount = np.exp(-r * dt)
    for i in range(N, 1, -1):
        # Il nuovo array C avrà 'i' elementi (riduzione di 1 elemento)
        C_nuovo = np.zeros(i)
        
        # Ciclo interno per calcolare ogni nodo nel tempo precedente
        # 'j' va da 0 a i-1 (indice del nodo al tempo precedente)
        for j in range(i):
            # C[j+1] è il valore in caso di UP-move dal nodo j
            # C[j] è il valore in caso di DOWN-move dal nodo j
            valore_atteso = p * C[j+1] + (1-p) * C[j]
            C_nuovo[j] = np.maximum(discount * valore_atteso,max(-ST_list[i-1][j]+K,0))
    
        
        # Sostituisci il vecchio array C con il nuovo array calcolato
        C = C_nuovo
    
    valore_atteso = p * C[1] + (1-p) * C[0]
    C = discount * valore_atteso

    
    # Dopo N iterazioni, C conterrà un solo elemento: il prezzo all'inizio (t=0)
    return C


#Leisen-Reimer model for an American call option
#----------------------------------------------------
def LR(S0, K, r, sigma, T, N):
    dt=T/N
    a=np.exp(r*dt)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    p=0.5+np.sqrt(0.25-0.25*np.exp(-((d2)/(N+(1/3)))**2*(N+(1/6))))
    P=0.5+np.sqrt(0.25-0.25*np.exp(-((d2+sigma*(np.sqrt(T)))/(N+(1/3)))**2*(N+(1/6))))
    u=a*(P/p)
    d=a*((1-P)/(1-p))

      # Building the stock tree
    ST_list=[S0]
    for j in range(1,N+1,1):
        box =[]
        for i in range(j+1):
         box.append(S0 * (u**i) * (d**(j-i)))
    
        ST_list.append(box)
    

    # option value at maturity 
    ST = np.array(ST_list[-1])
    C = np.maximum(-ST+K, 0.0)

    # backward induction
    discount = np.exp(-r * dt)
    for i in range(N, 1, -1):
        # Il nuovo array C avrà 'i' elementi (riduzione di 1 elemento)
        C_nuovo = np.zeros(i)
        
        # Ciclo interno per calcolare ogni nodo nel tempo precedente
        # 'j' va da 0 a i-1 (indice del nodo al tempo precedente)
        for j in range(i):
            # C[j+1] è il valore in caso di UP-move dal nodo j
            # C[j] è il valore in caso di DOWN-move dal nodo j
            valore_atteso = p * C[j+1] + (1-p) * C[j]
            C_nuovo[j] = np.maximum(discount * valore_atteso,max(-ST_list[i-1][j]+K,0))
    
        
        # Sostituisci il vecchio array C con il nuovo array calcolato
        C = C_nuovo
    
    valore_atteso = p * C[1] + (1-p) * C[0]
    C = discount * valore_atteso

    
    # Dopo N iterazioni, C conterrà un solo elemento: il prezzo all'inizio (t=0)
    return C


# SMOOTHING TECHNIQUES
# -------------------------

#Pegging the strike
#-------------------------------------------------
"""
def crr_PEG(S0, K, r, sigma, T, N):

    dt = T / N
    u=np.exp((sigma*np.sqrt(dt))+(dt*np.log(K/S0)))
    d=1/u
    p = (np.exp(r * dt) - d) / (u - d)

    ST_list=[S0]
    for j in range(1,N+1,1):
        box =[]
        for i in range(j+1):
         box.append(S0 * (u**i) * (d**(j-i)))
    
        ST_list.append(box)
    

    # option value at maturity 
    ST = np.array(ST_list[-1])
    C = np.maximum(K-ST, 0.0)

    # backward induction
    discount = np.exp(-r * dt)
    for i in range(N, 1, -1):
        # Il nuovo array C avrà 'i' elementi (riduzione di 1 elemento)
        C_nuovo = np.zeros(i)
        
        # Ciclo interno per calcolare ogni nodo nel tempo precedente
        # 'j' va da 0 a i-1 (indice del nodo al tempo precedente)
        for j in range(i):
            # C[j+1] è il valore in caso di UP-move dal nodo j
            # C[j] è il valore in caso di DOWN-move dal nodo j
            valore_atteso = p * C[j+1] + (1-p) * C[j]
            C_nuovo[j] = np.maximum(discount * valore_atteso,max(-ST_list[i-1][j]+K,0))
    
        
        # Sostituisci il vecchio array C con il nuovo array calcolato
        C = C_nuovo
    
    valore_atteso = p * C[1] + (1-p) * C[0]
    C = discount * valore_atteso

    
    # Dopo N iterazioni, C conterrà un solo elemento: il prezzo all'inizio (t=0)
    return C
"""

def crr_PEG(S0, K, r, sigma, T, N):
    dt = T / N

    u_crr = np.exp(sigma * np.sqrt(dt))
    d_crr = np.exp(-sigma * np.sqrt(dt))
    
   
    j_values = np.arange(N + 1)
    final_S_crr = S0 * (u_crr**(N - j_values)) * (d_crr**j_values)
    
    j_star = np.argmin(np.abs(final_S_crr - K))
    
   
    S_j_star = final_S_crr[j_star]
    gamma = (K / S_j_star)**(1 / N)

    
    u_peg = u_crr * gamma
    d_peg = d_crr * gamma
    
    # 5. Pegged Risk-Neutral Probability
    q_peg = (np.exp(r * dt) - d_peg) / (u_peg - d_peg)
    q_d = 1.0 - q_peg

   
    j_values = np.arange(N + 1)
    Sn = S0 * (u_peg**(N - j_values)) * (d_peg**j_values)
    
    V = np.maximum(-Sn + K, 0)
    
   
    for n in range(N - 1, -1, -1):
    
        V_expected = np.exp(-r * dt) * (q_peg * V[:n+1] + q_d * V[1:n+2])
        j_values_n = np.arange(n + 1)
        S_n = S0 * (u_peg**(n - j_values_n)) * (d_peg**j_values_n)
        
        V_intrinsic = np.maximum(-S_n + K, 0)
        V = np.maximum(V_intrinsic, V_expected)

    return V[0]


#Average smoothing
#------------------------------------------------
def crr_AS (S0, K, r, sigma, T, N):   # as = average smoothing

    p1 = crr(S0, K, r, sigma, T, N)
    p2 = crr(S0, K, r, sigma, T, N+1)

    return (p1+p2)/2


#Richardson Extrapolation
#--------------------------------------
def crr_RE (S0, K, r, sigma, T, N):

    p1 = crr(S0, K, r, sigma, T, N)
    p2 = crr(S0, K, r, sigma, T, 2*N)

    return (2*p2-p1)



#BlackScholes Smoothing
#-------------------------------------------------
"""
def crr_BSS(S0, K, r, sigma, T, N):# BSS = Black-Scholes smoothing
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    # Building the stock tree
    ST_list=[[S0]]
    for j in range(1,N+1,1):
        box =[]
        for i in range(j+1):
         box.append(S0 * (u**i) * (d**(j-i)))
    
        ST_list.append(box)
    

    # option value at maturity 
    ST = np.array(ST_list[-2])
    penult_vals=[]
    for i in range(len(ST)):
       Cont_vals = PEG_AS_RE(ST[i], K, r, sigma,T,50) 
       exerc_vals=max(-ST[i]+K, 0)
       penult_vals.append(max(Cont_vals, exerc_vals))
    
    C = penult_vals


    # backward induction
    discount = np.exp(-r * dt)
    for i in range(N-1, 0, -1):
        # Il nuovo array C avrà 'i' elementi (riduzione di 1 elemento)
        C_nuovo = np.zeros(i)
        
        # Ciclo interno per calcolare ogni nodo nel tempo precedente
        # 'j' va da 0 a i-1 (indice del nodo al tempo precedente)
        for j in range(i):
            # C[j+1] è il valore in caso di UP-move dal nodo j
            # C[j] è il valore in caso di DOWN-move dal nodo j
            valore_atteso = p * C[j+1] + (1-p) * C[j]
            C_nuovo[j] = np.maximum(discount * valore_atteso,-ST_list[i-1][j]+K)
    
        
        # Sostituisci il vecchio array C con il nuovo array calcolato
        C = C_nuovo
    
    # Dopo N iterazioni, C conterrà un solo elemento: il prezzo all'inizio (t=0)
    return C[0]

"""

#Richardson Extrapolation+BlackScholesSmoothing
#-------------------------------------------------
"""
def crr_BSS_RE (S0, K, r, sigma, T, N):

    p1 = crr_BSS(S0, K, r, sigma, T, N)
    p2 = crr_BSS(S0, K, r, sigma, T, 2*N)

    return (2*p2-p1)

"""

#Richardson Extrapolation + Average smoothing
#-------------------------------------------------
def crr_AS_RE (S0, K, r, sigma, T, N):

    p1 = crr_AS(S0, K, r, sigma, T, N)
    p2 = crr_AS(S0, K, r, sigma, T, 2*N)

    return (2*p2-p1)


#Richardson Extrapolation + Pegging the strike
#--------------------------------------------------
def crr_PEG_RE (S0, K, r, sigma, T, N):   

    p1 = crr_PEG(S0, K, r, sigma, T, N)
    p2 = crr_PEG(S0, K, r, sigma, T, N+1)
    
    return (p1+p2)/2


#BlackScholes Smoothing + Pegging the strike 
#--------------------------------------------------
"""
def crr_PEG_BSS(S0, K, r, sigma, T, N):
   dt = T / N
   u=np.exp((sigma*np.sqrt(dt))+(dt*np.log(K/S0)))
   d=1/u
   p = (np.exp(r * dt) - d) / (u - d)


   # Building the stock tree
   ST_list=[[S0]]
   for j in range(1,N+1,1):
        box =[]
        for i in range(j+1):
         box.append(S0 * (u**i) * (d**(j-i)))
    
        ST_list.append(box)
    

    # option value at maturity 
   ST = np.array(ST_list[-2])
   penult_vals=[]
   for i in range(len(ST)):
       Cont_vals =PEG_AS_RE(ST[i], K, r, sigma,T,50)
       exerc_vals=max(ST[i]-K, 0)
       penult_vals.append(max(Cont_vals, exerc_vals))
    
   C = penult_vals


    # backward induction
   discount = np.exp(-r * dt)
   for i in range(N-1, 0, -1):
        # Il nuovo array C avrà 'i' elementi (riduzione di 1 elemento)
        C_nuovo = np.zeros(i)
        
        # Ciclo interno per calcolare ogni nodo nel tempo precedente
        # 'j' va da 0 a i-1 (indice del nodo al tempo precedente)
        for j in range(i):
            # C[j+1] è il valore in caso di UP-move dal nodo j
            # C[j] è il valore in caso di DOWN-move dal nodo j
            valore_atteso = p * C[j+1] + (1-p) * C[j]
            C_nuovo[j] = np.maximum(discount * valore_atteso,ST_list[i-1][j]-K)
    
        
        # Sostituisci il vecchio array C con il nuovo array calcolato
        C = C_nuovo
    
    # Dopo N iterazioni, C conterrà un solo elemento: il prezzo all'inizio (t=0)
   return C[0]
   
"""

#BlackScholes Smoothing + Pegging the strike + Richardson Extrapolation
#-------------------------------------------------------------------------
"""
def crr_PEG_BSS_RE(S0, K, r, sigma, T, N):
   p1 = crr_PEG_BSS(S0, K, r, sigma, T, N)
   p2 = crr_PEG_BSS(S0, K, r, sigma, T, 2*N)

   return (2*p2-p1)

"""

#Leisen-Reimer + Richardson Extrapolation
#----------------------------------------------
def LR_RE (S0, K, r, sigma, T, N):

    p1 = LR(S0, K, r, sigma, T, N)
    p2 = LR(S0, K, r, sigma, T, 2*N+1)

    return (2*p2-p1)

# PEG+AS
def crr_PEG_AS (S0, K, r, sigma, T, N):   # as = average smoothing

    p1 = crr_PEG(S0, K, r, sigma, T, N)
    p2 = crr_PEG(S0, K, r, sigma, T, N+1)

    return (p1+p2)/2

#PEG+AS+RE

def PEG_AS_RE (S0, K, r, sigma, T, N):
    p1 = crr_PEG_AS(S0, K, r, sigma, T, N)
    p2 = crr_PEG_AS(S0, K, r, sigma, T, 2*N)

    return (2*p2-p1)



# BENCHMARK COMPUTATION
# -------------------------
# First parameters used
"""
N = 10000
BNK = crr(100,100,0.05,0.2,1,N)
print("BENCHMARK CRR COMPUTED (First parameters):",BNK)

BNK = LR_RE(100,100,0.05,0.2,1,7000)
print("BENCHMARK LR_RE COMPUTED (First parameters):",BNK)

BSP = black_scholes_put(100,100,0.05,0.2,1)
print("BENCHMARK BLACK-SCHOLES (First parameters):", BSP)
"""
#BNK = 6.090295412872714
# Parameters of the seminar
"""
N = 10000
BNK = crr(100,110,0.02,0.4,0.5,N)
print("BENCHMARK CRR COMPUTED:",BNK)
"""
BNK = LR_RE(100,110,0.02,0.4,0.5,7000)
print("BENCHMARK LR_RE COMPUTED:",BNK)

BSP = black_scholes_put(100,110,0.02,0.4,0.5)
print("BENCHMARK BLACK-SCHOLES:", BSP)



# Main comparison script
# -------------------------
def compare_models():
    """
    # Firts parameters used
    S0 = 100
    K = 100
    r = 0.05
    sigma = 0.2
    T = 1.0
    """
    # Parameters of the seminar
    S0 = 100
    K = 110
    r = 0.02
    sigma = 0.4
    T = 0.5
   

    # Different binomial steps
    steps = np.arange(25, 250, 3)
    l = len(steps)
    even_steps = steps[steps % 2 == 0]
    l_e = len(even_steps)
    odd_steps = steps[steps % 2 == 1]
    l_o =len(odd_steps)


    crr_p = np.zeros(l)
    tian_p = np.zeros(l)
    LR_p = np.zeros(l_o)
    crr_BSS_p = np.zeros(l)
    crr_AS_p = np.zeros(l)
    crr_PEG_p = np.zeros(l_e)
    crr_RE_p = np.zeros(l)
    crr_BSS_RE_p = np.zeros(l)
    crr_PEG_RE_p = np.zeros(l_e)
    crr_AS_RE_p = np.zeros(l)
    crr_PEG_BSS_p = np.zeros(l_e)
    crr_PEG_BSS_RE_p = np.zeros(l_e)
    LR_RE_p = np.zeros(l_o)

    PEG_AS_RE_p = np.zeros(l_e)
   
    

    index = 0
    index_e =0
    index_o =0
    # Compute prices values
    for N in steps:
        crr_p[index]= crr(S0, K, r, sigma, T, N)
        tian_p[index]=tian(S0, K, r, sigma, T, N)
        #crr_BSS_p[index] = crr_BSS(S0, K, r, sigma, T, N)
        crr_AS_p[index] = crr_AS(S0, K, r, sigma, T, N)
        crr_RE_p[index] = crr_RE(S0, K, r, sigma, T, N)
        crr_AS_RE_p[index] = crr_AS_RE(S0, K, r, sigma, T, N)
        #crr_BSS_RE_p[index] = crr_BSS_RE(S0, K, r, sigma, T, N)
        
        if N%2==0:
             crr_PEG_p[index_e] = crr_PEG(S0, K, r, sigma, T, N)
             crr_PEG_RE_p[index_e] = crr_PEG_RE(S0, K, r, sigma, T, N)
             #crr_PEG_BSS_p[index_e] = crr_PEG_BSS_RE(S0, K, r, sigma, T, N)
             #crr_PEG_BSS_RE_p[index_e] = crr_PEG_BSS_RE(S0, K, r, sigma, T, N)
             PEG_AS_RE_p[index_e] = PEG_AS_RE(S0, K, r, sigma, T, N)
             index_e+=1
        else:
            LR_p[index_o] = LR(S0, K, r, sigma, T, N)
            LR_RE_p[index_o] = LR_RE(S0, K, r, sigma, T, N)
            index_o+=1

        index +=1

    #errors
    crr_e = abs(crr_p - BNK)
    tian_e = abs(tian_p - BNK)
    LR_e = abs(LR_p - BNK)

    crr_BSS_e = abs(crr_BSS_p - BNK)
    crr_AS_e = abs(crr_AS_p - BNK)
    crr_PEG_e = abs(crr_PEG_p - BNK)
    crr_RE_e = abs(crr_RE_p - BNK)
    crr_BSS_RE_e = abs(crr_BSS_RE_p - BNK)
    crr_AS_RE_e = abs(crr_AS_RE_p - BNK)
    crr_PEG_RE_e = abs(crr_PEG_RE_p - BNK)
    crr_PEG_BSS_e = abs(crr_PEG_BSS_p - BNK)
    crr_PEG_BSS_RE_e = abs(crr_PEG_BSS_RE_p - BNK)

    PEG_AS_RE_e = abs(PEG_AS_RE_p - BNK)

    LR_RE_e = abs(LR_RE_p - BNK)


    
    
    # Price convergence
    # ----------------------------------------- #
    #CRR
    plt.figure(figsize=(12, 5))
    plt.plot(steps, crr_p , label="CRR price") 
    plt.axhline(BNK, linestyle="--", color = "black", label="Benchmark", alpha=0.7) 
    plt.title("CRR Convergence to Benchmark")
    plt.xlabel("Number of Steps (N)")
    plt.ylabel("American Put Price")
    plt.grid()
    plt.legend()


    #TIAN
    plt.figure(figsize=(12, 5))
    plt.plot(steps, tian_p , label="Tian price") 
    plt.axhline(BNK, linestyle="--", color = "black", label="Benchmark", alpha=0.7) 
    plt.title("Tian Convergence to Benchmark")
    plt.xlabel("Number of Steps (N)")
    plt.ylabel("American Put Price")
    plt.grid()
    plt.legend()
    
    #LR 
    plt.figure(figsize=(12, 5))
    plt.plot(odd_steps, LR_p, label="LR price")
    plt.axhline(BNK, linestyle="--", color = "black", label="Benchmark", alpha=0.7) 
    plt.title("LR Convergence to Benchmark")
    plt.xlabel("Number of Steps (N)")
    plt.ylabel("American Put Price")
    plt.grid()
    plt.legend()

    #ALL MODELS
    plt.figure(figsize=(12, 5))
    plt.plot(steps, tian_p, label="Tian price")
    plt.plot(odd_steps, LR_p, label="LR price")
    plt.plot(steps, crr_p , label="CRR price") 
    plt.axhline(BNK, linestyle="--", color = "black", label="Benchmark", alpha=0.7) 
    plt.title("CRR, Tian, LR Convergence to Benchmark")
    plt.xlabel("Number of Steps (N)")
    plt.ylabel("American Put Price")
    plt.grid()
    plt.legend()
 
 
    # Error plot
    # -------------------------------------------- #
    plt.figure(figsize=(12,5))
    plt.plot(steps, crr_e,  label="CRR")
    plt.plot(steps, 1/steps,  label="1/N")
    plt.title("CRR Absolute error")
    plt.xlabel("Number of Steps (N)")
    plt.ylabel("Error")
    plt.grid()
    plt.legend()


    plt.figure(figsize=(12,5))
    plt.plot(odd_steps,LR_e, label="LR")
    plt.plot(steps, 1/steps,  label="1/N")
    plt.title("LR Absolute error")
    plt.xlabel("Number of Steps (N)")
    plt.ylabel("Error")
    plt.grid()
    plt.legend()


    plt.figure(figsize=(12,5))
    plt.plot(steps,tian_e, label="Tian")
    plt.title("Tian Absolute error")
    plt.xlabel("Number of Steps (N)")
    plt.ylabel("Error")
    plt.grid()
    plt.legend()


    # Smoothing
    # -------------------------------------------- #
    plt.figure(figsize=(12,5))
    plt.plot(steps, crr_p , label="CRR price") 
    plt.plot(steps, crr_AS_p,  label="CRR + AS")
    plt.plot(even_steps, crr_PEG_p,  label="PEG")
    plt.title("Averaging Smoothing & Pegging the strike convergence")
    plt.axhline(BNK, linestyle="--", color = "black", label="Benchmark", alpha=0.7) 
    plt.xlabel("Number of Steps (N)")
    plt.ylabel("American Put Prices")
    plt.grid()
    plt.legend()

    #ERRORS
    plt.figure(figsize=(12,5))
    plt.plot(steps, crr_AS_e,  label="CRR + AS")
    plt.plot(even_steps, crr_PEG_e,  label="PEG")
    plt.title("Averaging Smoothing & Pegging the strike errors")
    plt.xlabel("Number of Steps (N)")
    plt.ylabel("Errors")
    plt.grid()
    plt.legend()
    
    # Richardson-Extrapolation for LE
    plt.figure(figsize=(12,5))
    plt.plot(odd_steps, LR_p,  label="LR")
    plt.plot(odd_steps, LR_RE_p,  label="LR + RE")
    plt.axhline(BNK, linestyle="--",color = "black", label="Benchmark", alpha=0.7) 
    plt.title("Richardson Extrapolation on CRR and LR models")
    plt.xlabel("Number of Steps (N)")
    plt.ylabel("prices")
    plt.grid()
    plt.legend()

    # error for LR + RE
    plt.figure(figsize=(12,5))
    plt.plot(odd_steps, LR_RE_e,  label="LR + RE")
    plt.title("LR + Richardson Extrapolation absolute error")
    plt.xlabel("Number of Steps (N)")
    plt.ylabel("Error")
    plt.grid()
    plt.legend()

    #Combination of techniques
    # -------------------------------------------- #
    plt.figure(figsize=(12, 5))
    plt.plot(odd_steps, LR_p, label="LR")
    plt.plot(odd_steps, LR_RE_p, label="LR + RE")
    plt.plot(steps, crr_p, label="CRR")
    plt.plot(steps, crr_AS_RE_p , label="CRR + AS + RE") 
    plt.plot(even_steps, crr_PEG_p, label="PEG")
    plt.plot(even_steps, PEG_AS_RE_p, label="PEG+AS+RE")
    plt.axhline(BNK, linestyle="--", color = "black", label="Benchmark", alpha=0.7) 
    plt.title("Convergence of different binomial models")
    plt.xlabel("Number of Steps (N)")
    plt.ylabel("American Put price")
    plt.grid()
    plt.legend()

    #ERROr plot
    plt.figure(figsize=(12, 5))
    plt.plot(odd_steps, LR_e, label="LR error")
    plt.plot(odd_steps, LR_RE_e, label="LR + RE error")
    plt.plot(steps, crr_e, label="CRR error")
    plt.plot(steps, crr_AS_RE_e , label="CRR + AS + RE error") 
    plt.plot(even_steps, crr_PEG_e, label="PEG error")
    plt.plot(even_steps, PEG_AS_RE_e, label="PEG + AS + RE error")
    plt.title("Errors of different binomial models")
    plt.xlabel("Number of Steps (N)")
    plt.ylabel("error")
    plt.grid()
    plt.legend()



    # last one
    plt.figure(figsize=(12, 5))
    plt.plot(odd_steps, LR_RE_p, label="LR + RE")
    plt.plot(steps, crr_AS_RE_p , label="CRR + AS + RE") 
    plt.plot(even_steps, PEG_AS_RE_p, label="PEG + AS + RE")
    plt.axhline(BNK, linestyle="--", color = "black", label="Benchmark", alpha=0.7) 
    plt.title("BEST METHODS Convergence to Benchmark")
    plt.xlabel("Number of Steps (N)")
    plt.ylabel("American Put price")
    plt.grid()
    plt.legend()

    #Error last one
    plt.figure(figsize=(12, 5))
    plt.plot(odd_steps, LR_RE_e, label="LR + RE error")
    plt.plot(steps, crr_AS_RE_e , label="CRR + AS + RE error") 
    plt.plot(even_steps, PEG_AS_RE_e, label="PEG+AS+RE")
    plt.title("BEST METHODS Errors")
    plt.xlabel("Number of Steps (N)")
    plt.ylabel("Error")
    plt.grid()
    plt.legend() 


    plt.tight_layout() 
    plt.show()


    # table of total errors

compare_models()