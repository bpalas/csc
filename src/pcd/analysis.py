import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse, find as sparse_find, spmatrix
import networkx as nx # --- NUEVO ---: Importar networkx
from .utils import _get_principal_eigenvector

def calculate_polarity(x_discrete, A_s_matrix):
    # ... (esta función no cambia) ...
    if not isinstance(x_discrete, np.ndarray): x_discrete = np.array(x_discrete)
    is_A_sparse = issparse(A_s_matrix)
    if not is_A_sparse and not isinstance(A_s_matrix, np.ndarray): A_s_matrix = np.array(A_s_matrix)
    if A_s_matrix.shape[0] != len(x_discrete) or A_s_matrix.shape[1] != len(x_discrete):
        if len(x_discrete) == 0 and A_s_matrix.shape == (0,0): return 0.0
        raise ValueError(f"Dimensiones incompatibles: A_s {A_s_matrix.shape}, x {len(x_discrete)}")
    if len(x_discrete) == 0: return 0.0
    numerator = x_discrete.T @ (A_s_matrix @ x_discrete) if is_A_sparse else x_discrete.T @ A_s_matrix @ x_discrete
    denominator = np.sum(x_discrete**2)
    if denominator < 1e-9: return 0.0
    return numerator / denominator


def calculate_edge_agreement_ratios(x_discrete, A_s_matrix):
    # ... (esta función no cambia) ...
    if x_discrete is None or len(x_discrete) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    S1_indices = np.where(x_discrete == 1)[0]
    S2_indices = np.where(x_discrete == -1)[0]
    num_pos_S1_internal, total_S1_internal = 0, 0
    num_pos_S2_internal, total_S2_internal = 0, 0
    num_neg_S1_S2_inter, total_S1_S2_inter = 0, 0
    if issparse(A_s_matrix):
        rows, cols, vals = sparse_find(A_s_matrix)
    else:
        rows_dense, cols_dense = np.where(np.abs(A_s_matrix) > 1e-9)
        vals = A_s_matrix[rows_dense, cols_dense]
        rows,cols = rows_dense, cols_dense
    for i_idx, j_idx, val_edge in zip(rows, cols, vals):
        if i_idx >= j_idx : continue
        xi, xj = x_discrete[i_idx], x_discrete[j_idx]
        if xi == 1 and xj == 1:
            total_S1_internal += 1
            if val_edge > 1e-9: num_pos_S1_internal += 1
        elif xi == -1 and xj == -1:
            total_S2_internal += 1
            if val_edge > 1e-9: num_pos_S2_internal += 1
        elif (xi == 1 and xj == -1) or (xi == -1 and xj == 1):
            total_S1_S2_inter += 1
            if val_edge < -1e-9: num_neg_S1_S2_inter += 1
    rpi_S1 = (num_pos_S1_internal / total_S1_internal) if total_S1_internal > 0 else 0.0
    rpi_S2 = (num_pos_S2_internal / total_S2_internal) if total_S2_internal > 0 else 0.0
    total_internal_edges = total_S1_internal + total_S2_internal
    total_pos_internal_edges = num_pos_S1_internal + num_pos_S2_internal
    rpi_combined = (total_pos_internal_edges / total_internal_edges) if total_internal_edges > 0 else 0.0
    rne_S1_S2 = (num_neg_S1_S2_inter / total_S1_S2_inter) if total_S1_S2_inter > 0 else 0.0
    components_present = 0; current_sum_agreement = 0.0
    if total_internal_edges > 0 :
        current_sum_agreement += rpi_combined; components_present +=1
    if total_S1_S2_inter > 0 :
        current_sum_agreement += rne_S1_S2; components_present +=1
    avg_agreement = (current_sum_agreement / components_present) if components_present > 0 else 0.0
    if len(S1_indices) == 0 and len(S2_indices) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    return rpi_S1, rpi_S2, rpi_combined, rne_S1_S2, avg_agreement


def calculate_objective_paper_k2(x_discrete_k2, A_s_matrix, alpha, beta):
    # ... (esta función no cambia) ...
    N_intra_pos, N_intra_neg, N_inter_pos, N_inter_neg = 0, 0, 0, 0
    size_s1 = np.sum(x_discrete_k2 == 1)
    size_s2 = np.sum(x_discrete_k2 == -1)

    rows, cols, vals = sparse_find(A_s_matrix)
    for i, j, v in zip(rows, cols, vals):
        if i >= j: continue
        xi, xj = x_discrete_k2[i], x_discrete_k2[j]

        if xi == xj and xi != 0: # Intra-cluster
            if v > 0: N_intra_pos += v
            else: N_intra_neg += abs(v)
        elif xi * xj < 0: # Inter-cluster (uno es 1 y otro -1)
            if v > 0: N_inter_pos += v
            else: N_inter_neg += abs(v)
            
    polarity_part = (N_intra_pos - N_intra_neg) + alpha * (N_inter_neg - N_inter_pos)
    regularization_part = beta * (size_s1**2 + size_s2**2)

    return polarity_part - regularization_part

def reduccion_dimensional(A_s_original: csr_matrix, df_nodos_original: pd.DataFrame,
                          node_to_idx_original: dict, eigen_solver_config: dict):
    # ... (esta función no cambia) ...
    print("--- Iniciando Reducción Dimensional al Núcleo del Conflicto ---")
    
    df_nucleo = df_nodos_original[df_nodos_original['CLUSTER_ASSIGNMENT'] != 0].copy()
    nodos_nucleo = df_nucleo['NODE_NAME'].tolist()
    m_nodes = len(nodos_nucleo)

    if m_nodes == 0:
        print("No hay nodos en el núcleo del conflicto para analizar.")
        columnas_esperadas = list(df_nodos_original.columns) + ['GRADO', 'V1_REPROYECTADO']
        return pd.DataFrame(columns=columnas_esperadas), csr_matrix((0, 0))

    indices_nucleo_original = [node_to_idx_original[n] for n in nodos_nucleo]
    
    A_p = A_s_original[np.ix_(indices_nucleo_original, indices_nucleo_original)]
    
    grados = np.abs(A_p).sum(axis=1).A1
    df_nucleo['GRADO'] = grados

    v1_reproyectado = _get_principal_eigenvector(
        A_p, m_nodes,
        eigen_solver=eigen_solver_config.get('eigen_solver', 'numpy_robust'),
        sparse_tol=eigen_solver_config.get('sparse_tol', 1e-3),
        v0_eigsh=eigen_solver_config.get('v0_eigsh', True),
        algo_name="REDUCCION_DIMENSIONAL"
    )
    df_nucleo['V1_REPROYECTADO'] = v1_reproyectado
    
    print(f"Reducción completada. El núcleo tiene {m_nodes} nodos.")

    df_resultado_ordenado = df_nucleo.sort_values(by="V1_REPROYECTADO", ascending=True).reset_index(drop=True)
    
    return df_resultado_ordenado, A_p

# --- NUEVO ---: Función para calcular el Coeficiente de Clustering Intra-Cluster
def calcular_coeficiente_clustering_intra_cluster(A_s: spmatrix, x_result: np.ndarray) -> dict:
    """
    Calcula el coeficiente de clustering promedio para los subgrafos
    formados por cada cluster.
    """
    # Para el clustering coefficient, no importa el signo, solo la existencia del enlace.
    A_unsigned = A_s.copy()
    A_unsigned.data = np.abs(A_unsigned.data)
    
    G = nx.from_scipy_sparse_array(A_unsigned, create_using=nx.Graph)
    
    nodos_idx_s1 = np.where(x_result == 1)[0]
    nodos_idx_s2 = np.where(x_result == -1)[0]

    G_s1 = G.subgraph(nodos_idx_s1)
    G_s2 = G.subgraph(nodos_idx_s2)

    # Se controla que el subgrafo tenga al menos 2 nodos para evitar errores.
    cc_s1 = nx.average_clustering(G_s1) if G_s1.number_of_nodes() > 1 else 0.0
    cc_s2 = nx.average_clustering(G_s2) if G_s2.number_of_nodes() > 1 else 0.0

    return {
        "CC_intra_S1": cc_s1,
        "CC_intra_S2": cc_s2
    }


# --- MODIFICADO ---: Se añade el parámetro 'calculate_intra_cluster_cc'
def calcular_metricas_paper(nombre_algoritmo: str, x_result: np.ndarray, A_s: csr_matrix, k: int, alpha: float, calculate_intra_cluster_cc: bool = False):
    """
    Calcula el diccionario de métricas completo según el Apéndice B.3 del paper.
    Opcionalmente, puede calcular y añadir el coeficiente de clustering intra-cluster.
    """
    if x_result is None or x_result.size == 0:
        return {"Algoritmo": nombre_algoritmo, "Error": "Resultado vacío"}

    # --- Cálculos de métricas existentes (sin cambios) ---
    nodos_s1 = np.where(x_result == 1)[0]
    nodos_s2 = np.where(x_result == -1)[0]
    n_s1, n_s2 = len(nodos_s1), len(nodos_s2)

    N_intra_pos, N_intra_neg, N_inter_pos, N_inter_neg, N_s0_inter = 0, 0, 0, 0, 0
    N_intra_pos_s1, N_intra_pos_s2 = 0, 0

    rows, cols, vals = sparse_find(A_s)
    for i, j, v in zip(rows, cols, vals):
        if i >= j: continue
        xi, xj = x_result[i], x_result[j]
        
        if xi == xj and xi != 0:
            if v > 0:
                N_intra_pos += v
                if xi == 1: N_intra_pos_s1 += v
                else: N_intra_pos_s2 += v
            else: N_intra_neg += abs(v)
        elif xi * xj < 0:
            if v > 0: N_inter_pos += v
            else: N_inter_neg += abs(v)
        elif (xi == 0 and xj != 0) or (xi != 0 and xj == 0):
            N_s0_inter += abs(v)
            
    SIZE = n_s1 + n_s2
    K_non_empty = (1 if n_s1 > 0 else 0) + (1 if n_s2 > 0 else 0)
    BAL = min(n_s1, n_s2) / max(n_s1, n_s2) if max(n_s1, n_s2) > 0 else 0.0
    
    denominador_pol = SIZE if SIZE > 0 else 1
    numerador_pol = (N_intra_pos - N_intra_neg) + alpha * (N_inter_neg - N_inter_pos)
    POL = numerador_pol / denominador_pol
    BA_POL = POL * BAL
    
    den_cc_plus = N_intra_pos + N_intra_neg
    CC_plus = (N_intra_pos - N_intra_neg) / den_cc_plus if den_cc_plus > 0 else 0.0
    den_cc_minus = N_inter_pos + N_inter_neg
    CC_minus = (N_inter_neg - N_inter_pos) / den_cc_minus if den_cc_minus > 0 else 0.0

    den_mac_s1 = n_s1 * (n_s1 - 1); den_mac_s2 = n_s2 * (n_s2 - 1)
    mac_s1 = N_intra_pos_s1 / den_mac_s1 if den_mac_s1 > 0 else 0.0
    mac_s2 = N_intra_pos_s2 / den_mac_s2 if den_mac_s2 > 0 else 0.0
    MAC = (mac_s1 + mac_s2) / k if k > 0 else 0.0
    
    den_mao = n_s1 * n_s2
    MAO = N_inter_neg / den_mao if den_mao > 0 else 0.0
    
    N_nz = N_intra_pos + N_intra_neg + N_inter_pos + N_inter_neg
    den_dens = SIZE * (SIZE - 1)
    DENS = N_nz / den_dens if den_dens > 0 else 0.0
    
    den_iso = N_nz + N_s0_inter
    ISO = N_nz / den_iso if den_iso > 0 else 0.0
    
    # --- Creación del diccionario de métricas ---
    metricas = {
        "Algoritmo": nombre_algoritmo, "POL": POL, "BA-POL": BA_POL, "SIZE": SIZE, 
        "BAL": BAL, "K": K_non_empty, "MAC": MAC, "MAO": MAO, "CC+": CC_plus, 
        "CC-": CC_minus, "DENS": DENS, "ISO": ISO
    }

    # --- NUEVO ---: Calcular y añadir las métricas de CC si se solicita
    if calculate_intra_cluster_cc:
        metricas_cc = calcular_coeficiente_clustering_intra_cluster(A_s, x_result)
        metricas.update(metricas_cc) # Fusiona las nuevas métricas al diccionario
    
    return metricas