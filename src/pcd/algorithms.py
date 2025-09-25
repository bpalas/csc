import numpy as np
from scipy.sparse import issparse
from scipy.sparse.linalg import eigsh

from .analysis import calculate_polarity 
from .utils import _get_principal_eigenvector
from .scg_algorithm import SCG # 👈 Importa la función principal de tu nuevo archivo


def run_eigensign(A_s_matrix, n_nodes_val, num_tau_candidates=100, eigen_solver='numpy_robust',
                  sparse_tol=1e-3, v0_eigsh=True):
    if n_nodes_val == 0: return np.array([]), -np.inf, 0.0, None # x, pol, tau, v1
    is_A_sparse = issparse(A_s_matrix)
    if (is_A_sparse and A_s_matrix.nnz == 0) or \
       (not is_A_sparse and ((A_s_matrix.size > 0 and np.all(np.isclose(A_s_matrix,0))) or A_s_matrix.size==0)):
        return np.zeros(n_nodes_val, dtype=int), 0.0, 0.0, np.zeros(n_nodes_val, dtype=float)

    v1_vector = _get_principal_eigenvector(A_s_matrix, n_nodes_val, eigen_solver, sparse_tol, v0_eigsh, "EIGENSIGN")
    if v1_vector is None: return np.zeros(n_nodes_val, dtype=int), -np.inf, 0.0, np.zeros(n_nodes_val, dtype=float)

    abs_v1 = np.abs(v1_vector)
    max_abs_v1 = np.max(abs_v1) if len(abs_v1) > 0 else 0.0
    tau_candidates = np.linspace(0, max_abs_v1, num_tau_candidates) if max_abs_v1 > 1e-9 else np.array([0.0])

    best_x, best_pol, best_tau = np.zeros(n_nodes_val, dtype=int), -np.inf, 0.0
    for tau in tau_candidates:
        x_temp = np.zeros(n_nodes_val, dtype=int); assign_idx = abs_v1 >= tau
        x_temp[assign_idx] = np.sign(v1_vector[assign_idx])
        # Asegurar que np.sign(0) resulte en 0, y que x_temp sea int
        x_temp = x_temp.astype(int) # np.sign puede devolver float

        pol = calculate_polarity(x_temp, A_s_matrix)
        if not np.isfinite(pol): pol = -np.inf
        if pol > best_pol: best_pol, best_x, best_tau = pol, x_temp.copy(), tau

    if not np.isfinite(best_pol): best_pol = 0.0
    return best_x, best_pol, best_tau, v1_vector

def run_random_eigensign(A_s_matrix, n_nodes_val, num_runs=100, eigen_solver='numpy_robust',
                         sparse_tol=1e-3, v0_eigsh=True, probability_scaler=1.0):
    if n_nodes_val == 0: return np.array([]), -np.inf, None # x, pol, v1
    is_A_sparse = issparse(A_s_matrix)
    if (is_A_sparse and A_s_matrix.nnz == 0) or \
       (not is_A_sparse and ((A_s_matrix.size > 0 and np.all(np.isclose(A_s_matrix,0))) or A_s_matrix.size==0)):
        return np.zeros(n_nodes_val, dtype=int), 0.0, np.zeros(n_nodes_val, dtype=float)

    v1_vector = _get_principal_eigenvector(A_s_matrix, n_nodes_val, eigen_solver, sparse_tol, v0_eigsh, "RANDOM-EIGENSIGN")
    if v1_vector is None: return np.zeros(n_nodes_val, dtype=int), -np.inf, np.zeros(n_nodes_val, dtype=float)

    best_x, best_pol = np.zeros(n_nodes_val, dtype=int), -np.inf
    norm_l1_v1 = np.sum(np.abs(v1_vector))
    if norm_l1_v1 < 1e-9 : norm_l1_v1 = 1.0 # Evitar división por cero

    for _ in range(num_runs):
        x_temp = np.zeros(n_nodes_val, dtype=int)
        for i in range(n_nodes_val):
            # Lógica original de probabilidad de la versión 2.0 del usuario
            prob_mag = probability_scaler * norm_l1_v1 * np.abs(v1_vector[i])
            prob = min(1.0, prob_mag) # Acotar probabilidad a [0,1]
            if np.random.rand() < prob: x_temp[i] = int(np.sign(v1_vector[i]))

        pol = calculate_polarity(x_temp, A_s_matrix)
        if np.isfinite(pol) and pol > best_pol: best_pol, best_x = pol, x_temp.copy()
        elif not np.isfinite(best_pol) and np.isfinite(pol): best_pol, best_x = pol, x_temp.copy() # Primer valor finito

    if not np.isfinite(best_pol): best_pol = 0.0
    return best_x, best_pol, v1_vector


# --- NUEVAS FUNCIONES PARA EL ALGORITMO DEL PAPER 2025 (k=2) ---

def run_local_search_paper_k2(A_s_matrix, n_nodes_val, **kwargs):
    """
    Implementación eficiente de la búsqueda local del paper de Aronsson et al. (2025)
    especializada para k=2 y con salida {-1, 0, 1} para compatibilidad.
    """
    # Hiperparámetros del algoritmo
    max_iter = kwargs.get('ls_max_iter', 20)
    # Por defecto, se usa la heurística alpha = 1/(k-1), que es 1 para k=2.
    alpha = kwargs.get('ls_alpha', 1.0) 
    # Beta es el hiperparámetro más importante para tunear.
    beta = kwargs.get('ls_beta', 0.1) 

    if n_nodes_val == 0:
        return np.array([]), 0.0

    # Inicialización aleatoria en {0, 1, 2} -> Neutral, Cluster 1, Cluster 2
    x_internal = np.random.randint(0, 3, size=n_nodes_val)
    
    # Pre-cómputo de la matriz M (Algoritmo 3 del paper)
    k = 2
    X = np.zeros((n_nodes_val, k))
    for i in range(n_nodes_val):
        if x_internal[i] > 0:
            X[i, x_internal[i]-1] = 1.0
    M = 2 * A_s_matrix @ X

    # Bucle de optimización
    for iter_num in range(max_iter):
        changed_in_iter = False
        node_indices = np.random.permutation(n_nodes_val)

        for i in node_indices:
            original_cluster = x_internal[i]
            
            # Cálculo del gradiente G (Ecuación 13 del paper)
            eta_i = np.sum(M[i, :])
            G = np.zeros(k + 1) # G[0], G[1], G[2]

            for m_idx in range(k): # m_idx=0 (C1), m_idx=1 (C2)
                cluster_label = m_idx + 1 # 1 o 2
                is_in_cluster = 1 if original_cluster == cluster_label else 0
                s_m_size = np.sum(x_internal == cluster_label)
                beta_im = 2 * beta * (s_m_size - is_in_cluster)
                
                G[cluster_label] = -beta + (1 + alpha) * M[i, m_idx] - beta_im - alpha * eta_i
            
            G[0] = 0.0 # Gradiente para el cluster neutral es siempre 0
            
            best_cluster = np.argmax(G)

            if best_cluster != original_cluster:
                changed_in_iter = True
                
                # Actualización eficiente de la matriz M
                A_i_col_dense = A_s_matrix[:, i].toarray().flatten() * 2
                if original_cluster > 0:
                    M[:, original_cluster - 1] -= A_i_col_dense
                if best_cluster > 0:
                    M[:, best_cluster - 1] += A_i_col_dense
                
                # Actualizar asignación del nodo
                x_internal[i] = best_cluster

        if not changed_in_iter and iter_num > 0:
            # print(f"Convergencia alcanzada en la iteración {iter_num+1}")
            break

    # Mapeo de la salida de {0, 1, 2} a {-1, 0, 1} para compatibilidad
    x_final = np.zeros(n_nodes_val, dtype=int)
    x_final[x_internal == 1] = 1
    x_final[x_internal == 2] = -1
    # Los que quedan en x_internal == 0 ya son 0 en x_final

    # Calcular la polaridad final para devolverla junto con las asignaciones
    final_polarity = calculate_polarity(x_final, A_s_matrix)
    
    return x_final, final_polarity

def run_scg(A_s_matrix, n_nodes_val, **kwargs):
    """
    Wrapper para el algoritmo SCG (Spectral Clustering for Graphs).
    Este algoritmo está diseñado para encontrar K comunidades.
    """
    print("--- Ejecutando SCG Algorithm ---")
    
    # 1. Obtener parámetros específicos de SCG desde la configuración (config)
    #    'K' es el número de comunidades, es un parámetro fundamental para SCG.
    #    El valor por defecto es 2 para que sea comparable con los otros algoritmos.
    K = kwargs.get('K', 2) 
    
    #    'rounding_strategy' es el método de redondeo que usa SCG.
    rounding_strategy = kwargs.get('rounding_strategy', 'min_angle')
    
    if K < 2:
        raise ValueError("SCG requiere K (número de comunidades) debe ser >= 2.")

    # 2. Llamar a la función SCG original
    #    Le pasamos la matriz de adyacencia (A) y el número de nodos (N).
    #    'dataset' es un parámetro dummy porque ya le estamos dando la matriz.
    C, _, _, _, _ = SCG(
        dataset='custom_graph', 
        K=K, 
        rounding_strategy=rounding_strategy, 
        N=n_nodes_val, 
        A=A_s_matrix
    )
    
    # 3. Adaptar la salida de SCG al formato de tu pipeline: {-1, 0, 1}
    #    SCG devuelve asignaciones en formato: -1 (neutral), 1 (cluster 1), 2 (cluster 2), ...
    #    Aquí lo adaptamos para K=2.
    x_final = np.zeros(n_nodes_val, dtype=int)
    if K == 2:
        x_final[C == 1] = 1   # Mapea cluster 1 a 1
        x_final[C == 2] = -1  # Mapea cluster 2 a -1
        # Los nodos con C == -1 (neutrales) ya son 0 en x_final.
    else:
        # Para K > 2, la correspondencia no es directa. Una opción es asignar
        # el primer cluster a '1' y todos los demás a '-1'.
        print(f"Advertencia: SCG con K={K}. Mapeando cluster 1 a (1), clusters 2..{K} a (-1).")
        x_final[C == 1] = 1
        x_final[(C > 1)] = -1
        
    # 4. Calcular la polaridad usando tu propia función para consistencia
    final_polarity = calculate_polarity(x_final, A_s_matrix)
    
    print(f"SCG finalizado. Polaridad calculada: {final_polarity:.4f}")
    
    return x_final, final_polarity