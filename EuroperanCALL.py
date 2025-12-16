import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math as m
from scipy.stats import norm

# Black-Scholes Formula for a European Call Option
def black_scholes_call(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call

# Defining Binomial Model
def crr(S0, K, r, sigma, T, N):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    # last layer of the stock tree
    ST_list=[]
    for j in range(N+1):
        ST_list.append(S0 * (u**j) * (d**(N - j)))
    ST=np.array(ST_list)

    # option value at maturity
    C = np.maximum(ST - K, 0.0)

    # backward induction
    discount = np.exp(-r * dt)
    for i in range(N, 0, -1):
        # Il nuovo array C avrà 'i' elementi (riduzione di 1 elemento)
        C_nuovo = np.zeros(i)
        
        # Ciclo interno per calcolare ogni nodo nel tempo precedente
        # 'j' va da 0 a i-1 (indice del nodo al tempo precedente)
        for j in range(i):
            # C[j+1] è il valore in caso di UP-move dal nodo j
            # C[j] è il valore in caso di DOWN-move dal nodo j
            valore_atteso = p * C[j+1] + (1-p) * C[j]
            C_nuovo[j] = discount * valore_atteso
        
        # Sostituisci il vecchio array C con il nuovo array calcolato
        C = C_nuovo
    
    # Dopo N iterazioni, C conterrà un solo elemento: il prezzo all'inizio (t=0)
    return C[0]

def tian(S0, K, r, sigma, T, N):
    dt = T / N
    M=np.exp(r*dt)
    V=np.exp(sigma**2*dt)
    u=0.5*(M*V)*(V+1+np.sqrt(V**2+2*V-3))
    d=0.5*(M*V)*(V+1-np.sqrt(V**2+2*V-3))
    p = (np.exp(r * dt) - d) / (u - d)

    # last layer of the stock tree
    ST_list=[]
    for j in range(N+1):
        ST_list.append(S0 * (u**j) * (d**(N - j)))
    ST=np.array(ST_list)


    # option value at maturity
    C = np.maximum(ST - K, 0.0)

    # backward induction
    discount = np.exp(-r * dt)
    for i in range(N, 0, -1):
        # Il nuovo array C avrà 'i' elementi (riduzione di 1 elemento)
        C_nuovo = np.zeros(i)
        
        # Ciclo interno per calcolare ogni nodo nel tempo precedente
        # 'j' va da 0 a i-1 (indice del nodo al tempo precedente)
        for j in range(i):
            # C[j+1] è il valore in caso di UP-move dal nodo j
            # C[j] è il valore in caso di DOWN-move dal nodo j
            valore_atteso = p * C[j+1] + (1-p) * C[j]
            C_nuovo[j] = discount * valore_atteso
        
        # Sostituisci il vecchio array C con il nuovo array calcolato
        C = C_nuovo
    
    # Dopo N iterazioni, C conterrà un solo elemento: il prezzo all'inizio (t=0)
    return C[0]

def LR(S0, K, r, sigma, T, N):
    dt=T/N
    a=np.exp(r*dt)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    p=0.5+np.sqrt(0.25-0.25*np.exp(-((d2)/(N+(1/3)))**2*(N+(1/6))))
    P=0.5+np.sqrt(0.25-0.25*np.exp(-((d2+sigma*(np.sqrt(T)))/(N+(1/3)))**2*(N+(1/6))))
    u=a*(P/p)
    d=a*((1-P)/(1-p))

    # last layer of the stock tree
    ST_list=[]
    for j in range(N+1):
        ST_list.append(S0 * (u**j) * (d**(N - j)))
    ST=np.array(ST_list)


    # option value at maturity
    C = np.maximum(ST - K, 0.0)

    # backward induction
    discount = np.exp(-r * dt)
    for i in range(N, 0, -1):
        # Il nuovo array C avrà 'i' elementi (riduzione di 1 elemento)
        C_nuovo = np.zeros(i)
        
        # Ciclo interno per calcolare ogni nodo nel tempo precedente
        # 'j' va da 0 a i-1 (indice del nodo al tempo precedente)
        for j in range(i):
            # C[j+1] è il valore in caso di UP-move dal nodo j
            # C[j] è il valore in caso di DOWN-move dal nodo j
            valore_atteso = p * C[j+1] + (1-p) * C[j]
            C_nuovo[j] = discount * valore_atteso
        
        # Sostituisci il vecchio array C con il nuovo array calcolato
        C = C_nuovo
    
    # Dopo N iterazioni, C conterrà un solo elemento: il prezzo all'inizio (t=0)
    return C[0]

# Defining Smoothing techniques and Richardson extrapolation
def crr_BSS(S0, K, r, sigma, T, N): # Black-Scholes smoothing
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    # penultimate layer of the stock tree
    ST_list=[]
    for j in range(N):
       S_p=S0 * (u**j) * (d**((N-1) - j))
       ST_list.append(black_scholes_call(S_p, K, r, sigma, dt))
    ST=np.array(ST_list)
    

    discount = np.exp(-r * dt)
    for i in range(N-1, 0, -1):
        C=np.zeros(i)
        for j in range(i):
            valore_atteso=p*ST[j+1]+(1-p)*ST[j]
            C[j]=discount*valore_atteso
        
        ST=C

    return ST[0]

def crr_AS(S0, K, r, sigma, T, N): # Averaging smoothing
    p1 = crr(S0, K, r, sigma, T, N)
    p2 = crr(S0, K, r, sigma, T, N+1)
    return (p1+p2)/2

def crr_RE (S0, K, r, sigma, T, N): # Richardson Extrapolation
    p1 = crr(S0, K, r, sigma, T, N)
    p2 = crr(S0, K, r, sigma, T, 2*N)
    return (2*p2-p1)

def LR_RE (S0, K, r, sigma, T, N): 
    p1 = LR(S0, K, r, sigma, T, N)
    p2 = LR(S0, K, r, sigma, T, 2*N+1)
    return ((4*p2-p1)/3)


def crr_PEG (S0, K, r, sigma, T, N): #Pegging the strike
    dt = T / N
    u=np.exp(sigma*np.sqrt(dt)+dt*np.log(K/S0))
    d=1/u
    p = (np.exp(r * dt) - d) / (u - d)

    # last layer of the stock tree
    ST_list=[]
    for j in range(N+1):
        ST_list.append(S0 * (u**j) * (d**(N - j)))
    ST=np.array(ST_list)

    # option value at maturity
    C = np.maximum(ST - K, 0.0)

    # backward induction
    discount = np.exp(-r * dt)
    for i in range(N, 0, -1):
        # Il nuovo array C avrà 'i' elementi (riduzione di 1 elemento)
        C_nuovo = np.zeros(i)
        
        # Ciclo interno per calcolare ogni nodo nel tempo precedente
        # 'j' va da 0 a i-1 (indice del nodo al tempo precedente)
        for j in range(i):
            # C[j+1] è il valore in caso di UP-move dal nodo j
            # C[j] è il valore in caso di DOWN-move dal nodo j
            valore_atteso = p * C[j+1] + (1-p) * C[j]
            C_nuovo[j] = discount * valore_atteso
        
        # Sostituisci il vecchio array C con il nuovo array calcolato
        C = C_nuovo
    
    # Dopo N iterazioni, C conterrà un solo elemento: il prezzo all'inizio (t=0)
    return C[0]

"""
def crr_PEG (S0, K, r, sigma, T, N): #Pegging the strike
    
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
    

    q_peg = (np.exp(r * dt) - d_peg) / (u_peg - d_peg)
    q_d = 1.0 - q_peg

  
    j_values = np.arange(N + 1)
    Sn = S0 * (u_peg**(N - j_values)) * (d_peg**j_values)
    

    V = np.maximum(Sn - K, 0)
    

    for n in range(N - 1, -1, -1):
      
        V = np.exp(-r * dt) * (q_peg * V[:n+1] + q_d * V[1:n+2])

    return V[0]
"""
def crr_BSS_RE (S0, K, r, sigma, T, N): 
    p1 = crr_BSS(S0, K, r, sigma, T, N)
    p2 = crr_BSS(S0, K, r, sigma, T, 2*N)
    return (2*p2-p1)

def crr_AS_RE (S0, K, r, sigma, T, N):
    p1 = crr_AS(S0, K, r, sigma, T, N)
    p2 = crr_AS(S0, K, r, sigma, T, 2*N)
    return (2*p2-p1)

def crr_PEG_RE (S0, K, r, sigma, T, N):
    p1 = crr_PEG(S0, K, r, sigma, T, N)
    p2 = crr_PEG(S0, K, r, sigma, T, 2*N)
    return (2*p2-p1)

def crr_PEG_BSS (S0, K, r, sigma, T, N):
    dt = T / N
    u=np.exp(sigma*np.sqrt(dt)+dt*np.log(K/S0))
    d=1/u
    p = (np.exp(r * dt) - d) / (u - d)

    # penultimate layer of the stock tree
    ST_list=[]
    for j in range(N):
       S_p=S0 * (u**j) * (d**((N-1) - j))
       ST_list.append(black_scholes_call(S_p, K, r, sigma, dt))
    ST=np.array(ST_list)
    

    discount = np.exp(-r * dt)
    for i in range(N-1, 0, -1):
        C=np.zeros(i)
        for j in range(i):
            valore_atteso=p*ST[j+1]+(1-p)*ST[j]
            C[j]=discount*valore_atteso
        
        ST=C

    return ST[0]

def crr_PEG_BSS_RE (S0, K, r, sigma, T, N):
    p1 = crr_PEG_BSS(S0, K, r, sigma, T, N)
    p2 = crr_PEG_BSS(S0, K, r, sigma, T, 2*N)
    return (2*p2-p1)

def crr_PEG_AS (S0, K, r, sigma, T, N):   # as = average smoothing

    p1 = crr_PEG(S0, K, r, sigma, T, N)
    p2 = crr_PEG(S0, K, r, sigma, T, N+1)

    return (p1+p2)/2

def PEG_AS_RE (S0, K, r, sigma, T, N):#PEG+AS+RE
    p1 = crr_PEG_AS(S0, K, r, sigma, T, N)
    p2 = crr_PEG_AS(S0, K, r, sigma, T, 2*N)

    return (2*p2-p1)

# -------------------------
# Main comparison script
# -------------------------
def compare_models():
    
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
    """

    # Different binomial steps
    steps = np.arange(200,400, 3)
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
    crr_PEG_AS_p = np.zeros(l_e)
   
    

    BSM_price = black_scholes_call(S0, K, r, sigma, T)
    index = 0
    index_e =0
    index_o =0
    # Compute prices values
    for N in steps:
        crr_p[index]= crr(S0, K, r, sigma, T, N)
        tian_p[index]=tian(S0, K, r, sigma, T, N)
        crr_BSS_p[index] = crr_BSS(S0, K, r, sigma, T, N)
        crr_AS_p[index] = crr_AS(S0, K, r, sigma, T, N)
        crr_RE_p[index] = crr_RE(S0, K, r, sigma, T, N)
        crr_AS_RE_p[index] = crr_AS_RE(S0, K, r, sigma, T, N)
        crr_BSS_RE_p[index] = crr_BSS_RE(S0, K, r, sigma, T, N)
        
        if N%2==0:
             crr_PEG_p[index_e] = crr_PEG(S0, K, r, sigma, T, N)
             crr_PEG_RE_p[index_e] = crr_PEG_RE(S0, K, r, sigma, T, N)
             crr_PEG_BSS_p[index_e] = crr_PEG_BSS_RE(S0, K, r, sigma, T, N)
             crr_PEG_BSS_RE_p[index_e] = crr_PEG_BSS_RE(S0, K, r, sigma, T, N)
             PEG_AS_RE_p[index_e] = PEG_AS_RE(S0, K, r, sigma, T, N)
             crr_PEG_AS_p[index_e] = crr_PEG_AS(S0, K, r, sigma, T, N)
             index_e+=1
        else:
            LR_p[index_o] = LR(S0, K, r, sigma, T, N)
            LR_RE_p[index_o] = LR_RE(S0, K, r, sigma, T, N)
            index_o+=1

        index +=1

    #errors
    crr_e = abs(crr_p - BSM_price)
    tian_e = abs(tian_p - BSM_price)
    LR_e = abs(LR_p - BSM_price)

    crr_BSS_e = abs(crr_BSS_p - BSM_price)
    crr_AS_e = abs(crr_AS_p - BSM_price)
    crr_PEG_e = abs(crr_PEG_p - BSM_price)
    crr_RE_e = abs(crr_RE_p - BSM_price)
    crr_BSS_RE_e = abs(crr_BSS_RE_p - BSM_price)
    crr_AS_RE_e = abs(crr_AS_RE_p - BSM_price)
    crr_PEG_RE_e = abs(crr_PEG_RE_p - BSM_price)
    crr_PEG_BSS_e = abs(crr_PEG_BSS_p - BSM_price)
    crr_PEG_BSS_RE_e = abs(crr_PEG_BSS_RE_p - BSM_price)

    LR_e = abs(LR_p -BSM_price)
    LR_RE_e = abs(LR_RE_p -BSM_price)
 
    # Price convergence
    # ----------------------------------------- #
    #CRR
    plt.figure(figsize=(12, 5))
    plt.plot(steps, crr_p , label="CRR price") 
    plt.axhline(BSM_price, linestyle="--", color="black",label="Black–Scholes", alpha=0.7) 
    plt.title("CRR Convergence to Black–Scholes")
    plt.xlabel("Number of Steps (N)")
    plt.ylabel("Call Price")
    plt.grid()
    plt.legend()

    #TIAN
    plt.figure(figsize=(12, 5))
    plt.plot(steps, tian_p, label="Tian price")
    plt.axhline(BSM_price, linestyle="--", color="black",label="Black–Scholes", alpha=0.7) 
    plt.title("Tian Convergence to Black–Scholes")
    plt.xlabel("Number of Steps (N)")
    plt.ylabel("Call Price")
    plt.grid()
    plt.legend()

    #LR
    plt.figure(figsize=(12, 5))
    plt.plot(odd_steps, LR_p , label="LR price") 
    plt.axhline(BSM_price, linestyle="--", color="black",label="Black–Scholes", alpha=0.7) 
    plt.title("LR method Convergence to Black–Scholes")
    plt.xlabel("Number of Steps (N)")
    plt.ylabel("Call Price")
    plt.grid()
    plt.legend()


    #ERRORS

    plt.figure(figsize=(12, 5))
    plt.plot(steps, tian_e, label="Tian error ")
    plt.title("Tian error European Call")
    plt.xlabel("Number of Steps (N)")
    plt.ylabel("Error")
    plt.grid()
    plt.legend()

    plt.figure(figsize=(12, 5))
    plt.plot(steps, crr_e, label="CRR error")
    plt.plot(steps, 1/steps, label="1/n")
    plt.title("CRR error European Call")
    plt.xlabel("Number of Steps (N)")
    plt.ylabel("Error")
    plt.grid()
    plt.legend()

    plt.figure(figsize=(12, 5))
    plt.plot(odd_steps, LR_e, label="LR error")
    plt.title("LR error European CALL")
    plt.xlabel("Number of Steps (N)")
    plt.ylabel("Error")
    plt.grid()
    plt.legend()
    
 
    # price comparison for all of the three models
    plt.figure(figsize=(12, 5))
    plt.plot(steps, tian_p, label="Tian price")
    plt.plot(odd_steps, LR_p, label="LR price")
    plt.plot(steps, crr_p , label="CRR price") 
    plt.axhline(BSM_price, linestyle="--", color = "black",label="Black–Scholes", alpha=0.7) 
    plt.title("CRR, Tian and LR Convergence to Black–Scholes")
    plt.xlabel("Number of Steps (N)")
    plt.ylabel("Call Price")
    plt.grid()
    plt.legend()

    #errors of the three models
    plt.figure(figsize=(12,5))
    plt.plot(steps, crr_e,  label="CRR")
    plt.plot(steps,tian_e, label="Tian")
    plt.plot(odd_steps,LR_e, label="LR")
    plt.title("Absolute error for the three models, european Call")
    plt.xlabel("Number of Steps (N)")
    plt.ylabel("Error")
    plt.grid()
    plt.legend()


    # Smoothing
    # -------------------------------------------- #

    # Black-Scholes smoothing
    plt.figure(figsize=(12,5))
    plt.plot(steps, crr_p , label="CRR price") 
    plt.plot(steps, crr_BSS_p,  label="CRR + BSS")
    plt.title("Black-Scholes Smoothing")
    plt.axhline(BSM_price, linestyle="--", color = "black",label="Black–Scholes", alpha=0.7) 
    plt.xlabel("Number of Steps (N)")
    plt.ylabel("prices")
    plt.grid()
    plt.legend()

    # Averaging smoothing
    plt.figure(figsize=(12,5))
    plt.plot(steps, crr_p , label="CRR price") 
    plt.plot(steps, crr_AS_p,  label="CRR + AS")
    plt.plot(even_steps, crr_PEG_AS_p,  label="PEG + AS")
    plt.title("Averaging Smoothing")
    plt.axhline(BSM_price, linestyle="--", color = "black",label="Black–Scholes", alpha=0.7) 
    plt.xlabel("Number of Steps (N)")
    plt.ylabel("prices")
    plt.grid()
    plt.legend()

    #Pegging the strike
    plt.figure(figsize=(12,5))
    plt.plot(even_steps, crr_PEG_p,  label="CRR + PEG")
    plt.plot(steps, crr_p , label="CRR price") 
    plt.title("Pegging the strike")
    plt.axhline(BSM_price, linestyle="--", color = "black",label="Black–Scholes", alpha=0.7) 
    plt.xlabel("Number of Steps (N)")
    plt.ylabel("prices")
    plt.grid()
    plt.legend()



    # Richardson-Extrapolation
    plt.figure(figsize=(12,5))
    plt.plot(odd_steps, LR_p,  label="LR ")
    #plt.plot(odd_steps, LR_RE_p,  label="LR + RE")
    plt.axhline(BSM_price, linestyle="--", color = "black",label="Black–Scholes", alpha=0.7) 
    plt.title("Richardson Extrapolation")
    plt.xlabel("Number of Steps (N)")
    plt.ylabel("prices")
    plt.grid()
    plt.legend()

    #Combination of techniques
    # -------------------------------------------- #
    plt.figure(figsize=(12, 5))
    plt.plot(odd_steps, LR_p, label="LR")
    plt.plot(odd_steps, LR_RE_p, label="LR + RE")
    plt.plot(steps, crr_p, label="CRR")
    plt.plot(steps, crr_BSS_p , label="CRR + BSS") 
    plt.plot(steps, crr_AS_RE_p , label="CRR + AS + RE") 
    plt.plot(even_steps, crr_PEG_p, label="PEG")
    plt.plot(even_steps, crr_PEG_BSS_p, label="PEG + BSS")
    plt.plot(even_steps, crr_PEG_BSS_RE_p, label="PEG + BSS + RE")
    plt.plot(even_steps, PEG_AS_RE_p, label="PEG + AS+ RE")
    plt.axhline(BSM_price, linestyle="--", color = "black",label="Black–Scholes", alpha=0.7) 
    plt.title("Combination of Techniques Convergence to Black–Scholes")
    plt.xlabel("Number of Steps (N)")
    plt.ylabel("Call Price")
    plt.grid()
    plt.legend()

    # last one
    plt.figure(figsize=(12, 5))
    plt.plot(odd_steps, LR_p, label="LR ")
    plt.plot(odd_steps, LR_RE_p, label="LR + RE")
    plt.plot(steps, crr_AS_RE_p , label="CRR + AS + RE") 
    plt.plot(even_steps, crr_PEG_BSS_RE_p, label="PEG + BSS + RE")
    plt.plot(even_steps, PEG_AS_RE_p, label="PEG + AS+ RE")
    plt.axhline(BSM_price, linestyle="--", color = "black",label="Black–Scholes", alpha=0.7) 
    plt.title("BEST METHODS Convergence to Black–Scholes")
    plt.xlabel("Number of Steps (N)")
    plt.ylabel("Call Price")
    plt.grid()
    plt.legend()

    plt.tight_layout() 
    plt.show()
   
    # table of total errors


compare_models()

    
