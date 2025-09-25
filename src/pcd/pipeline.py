
import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np

# --- 1. Asegúrate de que los imports estén correctos ---
from .data_processing import preprocess_and_create_signed_adjacency_matrix
from .algorithms import run_eigensign, run_random_eigensign, run_local_search_paper_k2, run_scg
from .analysis import (
    calculate_edge_agreement_ratios,
    calculate_objective_paper_k2,
    calcular_metricas_paper, 
)
from .visualization import clivaje_recta


def ejecutar_analisis_polarizacion(
    df_input: pd.DataFrame,
    config: dict,
    default_cols: dict,
    calculate_intra_cluster_cc: bool = False
):
    print(f"--- Iniciando Análisis: {config.get('name', 'Sin Nombre')} ---")

    A_s, node_to_idx, idx_to_node, n_nodes, _ = \
        preprocess_and_create_signed_adjacency_matrix(
            df_input=df_input,
            from_node_col=default_cols.get('from_node_col', 'FROM_NODE'),
            to_node_col=default_cols.get('to_node_col', 'TO_NODE'),
            sign_col=default_cols.get('sign_col', 'SIGN'),
            node_norm_strategy=default_cols.get("node_norm_strategy", "lower_unidecode_strip"),
            weighting_strategy=default_cols.get("weighting_strategy", "binary_sum_signs_actual")
        )

    if n_nodes == 0:
        print("Advertencia: El grafo no tiene nodos. Devolviendo resultados vacíos.")
        empty_metrics = {"Experimento": config.get('name', 'Sin Nombre'), "Algoritmo": config.get("algorithm_type"), "Polaridad": 0}
        empty_paper_metrics = {"Algoritmo": config.get('name', 'Sin Nombre'), "Error": "Grafo vacío"}
        return pd.DataFrame(columns=['NODE_NAME', 'CLUSTER_ASSIGNMENT']), empty_metrics, empty_paper_metrics, csr_matrix((0,0)), {}

    algorithm = config.get("algorithm_type")
    print(f"Ejecutando algoritmo de clustering: {algorithm}")

    algo_config = config.copy()
    algo_config.pop('name', None)
    algo_config.pop('algorithm_type', None)

    cluster_assignments = np.zeros(n_nodes, dtype=int)
    polarity = 0.0
    extra_metrics = {}

    if algorithm == "eigensign":
        cluster_assignments, polarity, _, _ = run_eigensign(A_s, n_nodes, **algo_config)
    elif algorithm == "random_eigensign":
        cluster_assignments, polarity, _ = run_random_eigensign(A_s, n_nodes, **algo_config)
    elif algorithm == "local_search_paper_k2":
        cluster_assignments, polarity = run_local_search_paper_k2(A_s, n_nodes, **algo_config)
        alpha = algo_config.get('ls_alpha', 1.0)
        beta = algo_config.get('ls_beta', 0.1)
        objective_val = calculate_objective_paper_k2(cluster_assignments, A_s, alpha, beta)
        extra_metrics['Objective_Paper2025'] = objective_val
    elif algorithm == "scg":
        cluster_assignments, polarity = run_scg(A_s, n_nodes, **algo_config)
    else:
        raise ValueError(f"Algoritmo de clustering desconocido: '{algorithm}'")
    
    if cluster_assignments is None: 
        cluster_assignments = np.zeros(n_nodes, dtype=int)

    # --- INICIO DE LA MODIFICACIÓN ---

    # 2. Calcular las métricas originales
    print("Calculando métricas originales...")
    rpi_s1, rpi_s2, rpi_comb, rne_s1s2, avg_agree = calculate_edge_agreement_ratios(cluster_assignments, A_s)
    
    s1_c = np.sum(cluster_assignments == 1)
    s2_c = np.sum(cluster_assignments == -1)
    s0_c = n_nodes - (s1_c + s2_c)
    s1s2_ratio = (s1_c + s2_c) / float(n_nodes) if n_nodes > 0 else 0.0

    core_metrics = {
        "Experimento": config.get('name', 'Sin Nombre'),
        "Algoritmo": algorithm,
        "Polaridad": polarity,
        "RPI_S1": rpi_s1, "RPI_S2": rpi_s2, "RPI_comb": rpi_comb,
        "RNE_S1S2": rne_s1s2, "AvgAgree": avg_agree,
        "N_S1": s1_c, "N_S2": s2_c, "N_S0": s0_c,
        "N_Total": n_nodes, "$S_1+S_2$ Ratio": s1s2_ratio,
        **extra_metrics
    }

    # 3. Calcular las nuevas métricas del paper
    print("Calculando métricas del paper...")
    k = config.get('k', 2)
    # Usa un valor de alpha razonable, puedes obtenerlo del config o usar un default
    alpha = config.get('ls_alpha', 1.0 / (k - 1) if k > 1 else 1.0)
    
    paper_metrics = calcular_metricas_paper(
        nombre_algoritmo=config.get('name', 'Sin Nombre'),
        x_result=cluster_assignments,
        A_s=A_s,
        k=k,
        alpha=alpha,
        calculate_intra_cluster_cc=calculate_intra_cluster_cc # <--- AÑADE ESTO

    )

    node_details = [{"NODE_NAME": idx_to_node[i], "CLUSTER_ASSIGNMENT": cluster_assignments[i]} for i in range(n_nodes)]
    df_nodes_results = pd.DataFrame(node_details)
    
    print("--- Análisis completado. ---")

    # 4. Devuelve ambos diccionarios de métricas
    return df_nodes_results, core_metrics, paper_metrics, A_s, node_to_idx

    # --- FIN DE LA MODIFICACIÓN ---