# scg_algorithm.py
# Contiene la implementación completa del algoritmo SCG y sus funciones de utilidad.
# Este módulo está diseñado para ser importado y utilizado por otros scripts.

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, norm
from scipy.stats import mode
# from blist import sorteddict  <-- LÍNEA ELIMINADA
import time

# ==============================================================================
# SECCIÓN 1: LÓGICA PRINCIPAL DEL ALGORITMO SCG
# ==============================================================================

def SCG(dataset, K, rounding_strategy, N=None, A=None):
    """ Encuentra K comunidades polarizadas. """
    if dataset != 'sbm': # Conjunto de datos del mundo real
        print('------ Ejecutando para el grafo proporcionado ------')
        if N is None or A is None:
            raise ValueError("Para grafos personalizados, se deben proporcionar N (n_nodes) y A (adjacency matrix).")
    else: # SBM sintético modificado
        pass
    
    # Descomposición propia de la matriz núcleo KI-1
    D,U = EigenDecompose_Core(K)
    U = U[:, D.argsort()]
    D = np.sort(D)
    
    # Inicialización
    Y = np.zeros((N,K))
    C = np.array([-1 for i in range(N)]) # Asignación de clúster, C_i en {-1, 1, ..., K}
    mask = np.ones((N)) # Lista de nodos a asignar
    maskA = A.copy() # Matriz de adyacencia del grafo restante

    for z in reversed(range(1,K)): # Asignar desde el 1er, ..., (K-1)-ésimo, K-ésimo clúster
        try:
            # Evitar error en grafos muy pequeños o disconexos
            if maskA.nnz == 0:
                print(f"Advertencia: No quedan aristas en la iteración z={z}. Deteniendo el proceso.")
                break
            lD, lU = eigsh(maskA, k=1, which='LA') # Vector propio del valor propio más grande
            sD, _ = eigsh(maskA, k=1, which='SA') # Vector propio del valor propio más pequeño
            lD, sD = lD[0], sD[0]
            v = lU[:,0].reshape((-1))
        except Exception as e:
            print(f"Advertencia: eigsh falló en la iteración z={z}. Puede que el grafo restante esté vacío o desconectado. {e}")
            break # Salir si no se pueden calcular los vectores propios

        zi = K-z # Esta iteración decide el clúster zi
        
        # Redondear v a {-1,0,z}^n
        if rounding_strategy=='min_angle':
            v_round = round_by_min_angle(v, z, -1, mask, N)
        elif rounding_strategy=='randomized':
            v_round = round_by_randomized_vector(v, z, -1, mask, maskA, N)
        elif rounding_strategy=='max_obj':
            v_round = round_by_max_obj_one_threshold(v, z, -1, mask, maskA, N)
        elif rounding_strategy=='bansal':
            v_round = round_by_cc_bansal(z, -1, mask, maskA, N)
        else:
            raise ValueError(f"Estrategia de redondeo desconocida: {rounding_strategy}")
            
        # Asignar al(los) nuevo(s) clúster(es)
        for i in range(N):
            if v_round[i]==0: continue
            if z>1:
                if v_round[i]>0: C[i], Y[i,:] = zi, U[zi-1,:].copy() # Asignar al clúster zi
            else:
                if v_round[i]>0: C[i], Y[i,:] = zi, U[zi-1,:].copy() # Asignar al clúster (K-1)
                else: C[i], Y[i,:] = zi+1, U[zi,:].copy() # Asignar al clúster K
                
        # Comprobar el valor objetivo actual
        print('{}-ésima iteración obj={:.1f}, x^TAx/x^Tx={:.1f} en ({:.1f}, {:.1f})'.format(
            zi, compute_Obj(Y, A, K), compute_RayleighsQuotient(v_round, maskA), sD, lD))
            
        # Establecer los nodos asignados para ser omitidos en la siguiente iteración
        for i in range(N):
            if v_round[i]>0:
                # Eliminar todas las aristas incidentes a los nodos asignados
                maskA[i,:] = maskA[i,:].multiply(0)
                maskA[:,i] = maskA[:,i].multiply(0)
                mask[i] = 0 # Eliminar el nodo asignado de la lista restante
                
    return C, Y, A, N, K

# ==============================================================================
# SECCIÓN 2: FUNCIONES DE UTILIDAD REQUERIDAS POR SCG
# ==============================================================================

# Constantes
EPS=1E-10
INF=1E10

def EigenDecompose_Core(K):
    """ Devuelve el espectro de la matriz (KI-1_{KxK}). """
    U = [[np.sqrt(1.0/K) for i in range(K)]] # Vector propio
    D = [2-K] # Valor propio
    for i in range(K-1):
        x = []
        for j in range(i): x += [0]
        x += [-(K-1-i)*np.sqrt(1.0/K)]
        for j in range(K-i-1): x += [np.sqrt(1.0/K)]
        s = np.sqrt(sum([j*j for j in x]))
        x = [j/s for j in x]
        U += [x]
        D += [2]
    return np.array(D), np.array(U).T

def compute_Obj(Y, A, K):
    """ sum_{j=2}^K de (Y_{:,j})^TA(Y_{:,j}) / ((Y_{:,j})^T(Y_{:,j})). """
    num, de = 0, 0
    if Y.shape[1] < K: return 0.0 # Evitar error si Y no tiene suficientes columnas
    for i in range(K-1):
        num += (Y[:,i+1].T).dot(A.dot(Y[:,i+1]))
        de += Y[:,i+1].T @ Y[:,i+1]
    if de < EPS: return 0.0
    return (num / de)

def compute_RayleighsQuotient(Y, A):
    """ Cociente de Rayleigh con el vector Y y la matriz de entrada A. """
    denominator = Y.T @ Y
    if denominator < EPS: return 0.0
    return (Y.T) @ (A.dot(Y)) / denominator

# ==============================================================================
# SECCIÓN 3: ALGORITMOS DE REDONDEO
# ==============================================================================

#### [Determinista] Ángulo Mínimo ####
def min_angle_find_k1_k2(v, idx, pos, neg, N):
    def next_move(v, T, i, j):
        distances = []
        for ci,cj in [(1,0),(0,-1)]:
            cT = T.copy()
            ti, tj = i+ci, j+cj
            i_invalid, j_invalid = (ti<0 or ti>=len(idx)), (tj<0 or tj>=len(idx))
            if i_invalid and j_invalid: continue
            if i_invalid and cj==0: continue
            if j_invalid and ci==0: continue
            if not i_invalid and ci==1: cT[idx[ti]] = pos
            elif not j_invalid and cj==-1: cT[idx[tj]] = neg
            
            norm_cT = np.linalg.norm(cT)
            if norm_cT < EPS: continue
            
            e = cT/norm_cT
            diff = np.linalg.norm(v - v.dot(e)*e)
            distances += [(ti,tj,diff,cT.copy())]
        if not distances:
            return i, j, INF, T
        min_idx = np.argmin([x[2] for x in distances])
        return distances[min_idx]
    k1, k2, dist_opt, T_opt = -1, N, INF, np.zeros((N))
    while k1<k2:
        k1_next, k2_next, dist_next, T_next = next_move(v, T_opt, k1, k2)
        if dist_next >= dist_opt: break
        k1, k2, dist_opt, T_opt = k1_next, k2_next, dist_next, T_next.copy()
    return T_opt, dist_opt, k1, k2

def round_by_min_angle(v, pos, neg, mask, N):
    v = v*mask
    idx_x = np.argsort(v, axis=0)[::-1]
    idx_y = idx_x[::-1]
    x, x_diff, _, _ = min_angle_find_k1_k2(v, idx_x, pos, neg, N)
    y, y_diff, _, _ = min_angle_find_k1_k2(-v, idx_y, pos, neg, N)
    v_round = x if x_diff < y_diff else y
    return v_round

#### [Determinista] Objetivo Máximo ####
def round_by_max_obj_one_threshold(v_in, pos, neg, mask, A, N):
    def max_obj_find_th(v):
        T_opt, obj_opt = np.zeros(N), -INF
        unique_thresholds = set(np.abs(v))
        for th in unique_thresholds:
            if th < EPS: continue
            T = pos*(v>th) + neg*(v<-th)
            if np.sum(np.abs(T))==0: continue
            obj = compute_RayleighsQuotient(T, A)
            if obj > obj_opt: T_opt, obj_opt = T.copy(), obj
        return T_opt, obj_opt
    v_in = v_in*mask
    x, x_obj = max_obj_find_th(v_in)
    y, y_obj = max_obj_find_th(-v_in)
    v_round = x if x_obj > y_obj else y
    return v_round.copy()

#### [Aleatorizado] ####
def round_by_randomized_vector(v_in, pos, neg, mask, A, N):
    def randomized_vector(v):
        def bernoulli_sample(x):
            if x>0: return pos*np.random.choice([0,1], 1, p=[max(1-x/pos,0),min(x/pos,1)])[0]
            elif x<0: return neg*np.random.choice([0,1], 1, p=[max(1-x/neg,0),min(x/neg,1)])[0]
            else: return 0
        v_scaled = v * np.sum(np.abs(v))
        T = np.array([bernoulli_sample(vi) for vi in v_scaled])
        return T
    v_in = v_in*mask
    x = randomized_vector(v_in)
    y = randomized_vector(-v_in)
    v_round = x.copy() if compute_RayleighsQuotient(x, A) > compute_RayleighsQuotient(y, A) else y.copy()
    return v_round

#### [Correlation Clustering: Bansal 3-approximation] ####
def round_by_cc_bansal(pos, neg, mask, A, N):
    def find_one_neighborhood_split():
        T_opt, obj_opt = np.zeros(N), -INF
        active_nodes = np.where(mask > 0)[0]
        for i in active_nodes:
            _, nbrs = A[i,:].nonzero()
            S1, S2 = [], []
            for j in nbrs:
                if mask[j] == 0: continue # Solo considerar vecinos no asignados
                if A[i,j]>0: S1.append(j)
                elif A[i,j]<0: S2.append(j)
            
            T1, T2 = np.zeros(N), np.zeros(N)
            for j in S1: T1[j], T2[j] = pos, neg
            for j in S2: T1[j], T2[j] = neg, pos
            
            obj1 = compute_RayleighsQuotient(T1, A)
            obj2 = compute_RayleighsQuotient(T2, A)
            
            T_i, obj_i = (T1, obj1) if obj1 > obj2 else (T2, obj2)
            if obj_i > obj_opt:
                T_opt, obj_opt = T_i.copy(), obj_i
        return T_opt
    v_round = find_one_neighborhood_split()
    return v_round