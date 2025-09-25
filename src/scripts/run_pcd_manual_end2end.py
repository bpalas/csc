import os
import sys
import glob
import argparse
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from pcd.data_processing import preprocess_and_create_signed_adjacency_matrix
from pcd.algorithms import run_local_search_paper_k2


def split_signed(A_s: csr_matrix) -> Tuple[csr_matrix, csr_matrix, csr_matrix]:
    A = A_s.tocoo(copy=True)
    pos_mask = A.data > 0
    neg_mask = A.data < 0
    A_pos = csr_matrix((A.data[pos_mask], (A.row[pos_mask], A.col[pos_mask])), shape=A.shape)
    A_neg = csr_matrix((np.abs(A.data[neg_mask]), (A.row[neg_mask], A.col[neg_mask])), shape=A.shape)
    A_abs = csr_matrix((np.abs(A.data), (A.row, A.col)), shape=A.shape)
    A_pos.sum_duplicates(); A_pos.eliminate_zeros()
    A_neg.sum_duplicates(); A_neg.eliminate_zeros()
    A_abs.sum_duplicates(); A_abs.eliminate_zeros()
    return A_pos, A_neg, A_abs


def symmetric_normalize(A: csr_matrix) -> csr_matrix:
    A = A.tocsr(copy=True)
    deg = np.asarray(A.sum(axis=1)).ravel()
    with np.errstate(divide='ignore'):
        inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    inv_sqrt[~np.isfinite(inv_sqrt)] = 0.0
    D_left = inv_sqrt
    A = A.tocoo()
    data = A.data * D_left[A.row] * D_left[A.col]
    return csr_matrix((data, (A.row, A.col)), shape=A.shape)


def normalize_vec(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    mn = np.nanmin(x)
    mx = np.nanmax(x)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return np.zeros_like(x, dtype=float)
    return (x - mn) / (mx - mn)


def power_iteration(A: csr_matrix, iters: int = 50, eps: float = 1e-12) -> np.ndarray:
    n = A.shape[0]
    if n == 0 or A.nnz == 0:
        return np.zeros(n, dtype=float)
    x = np.ones(n, dtype=float) / np.sqrt(max(1, n))
    for _ in range(max(1, iters)):
        x_new = A @ x
        norm = np.linalg.norm(x_new)
        if norm <= eps:
            return np.zeros(n, dtype=float)
        x = x_new / norm
    return x


def compute_partition_metrics(A_pos: csr_matrix, A_neg: csr_matrix, A_abs: csr_matrix, labels: np.ndarray) -> Dict[str, np.ndarray]:
    n = A_pos.shape[0]
    lab = labels.astype(int)

    # Precompute sets by label
    S1 = set(np.where(lab == 1)[0].tolist())
    S2 = set(np.where(lab == -1)[0].tolist())
    S0 = set(np.where(lab == 0)[0].tolist())

    # Initialize accumulators
    d_intra_pos = np.zeros(n, dtype=float)
    d_intra_neg = np.zeros(n, dtype=float)
    d_inter_pos = np.zeros(n, dtype=float)
    d_inter_neg = np.zeros(n, dtype=float)
    d_neut_pos = np.zeros(n, dtype=float)
    d_neut_neg = np.zeros(n, dtype=float)
    d_S1 = np.zeros(n, dtype=float)
    d_S2 = np.zeros(n, dtype=float)
    d_total = np.asarray(A_abs.sum(axis=1)).ravel().astype(float)

    Apos = A_pos.tocsr(); Aneg = A_neg.tocsr()
    for u in range(n):
        ru_p = slice(Apos.indptr[u], Apos.indptr[u+1])
        cols_p = Apos.indices[ru_p]
        vals_p = Apos.data[ru_p]
        ru_n = slice(Aneg.indptr[u], Aneg.indptr[u+1])
        cols_n = Aneg.indices[ru_n]
        vals_n = Aneg.data[ru_n]
        cu = lab[u]
        # positive edges
        for v, w in zip(cols_p, vals_p):
            lv = lab[v]
            if cu != 0 and lv == cu:
                d_intra_pos[u] += w
            elif cu != 0 and lv == -cu:
                d_inter_pos[u] += w
            else:
                d_neut_pos[u] += w
            if lv == 1:
                d_S1[u] += w
            elif lv == -1:
                d_S2[u] += w
        # negative edges
        for v, w in zip(cols_n, vals_n):
            lv = lab[v]
            if cu != 0 and lv == cu:
                d_intra_neg[u] += w
            elif cu != 0 and lv == -cu:
                d_inter_neg[u] += w
            else:
                d_neut_neg[u] += w
            if lv == 1:
                d_S1[u] += w
            elif lv == -1:
                d_S2[u] += w

    # S_eq
    denom = d_S1 + d_S2
    S_eq = np.where(denom > 0, (d_S1 - d_S2) / denom, 0.0)

    # Anomalía fronteriza
    d_anom = d_inter_pos.copy()
    d_inter = d_inter_pos + d_inter_neg
    P_anom = np.where(d_inter > 0, d_inter_pos / d_inter, 0.0)

    # Exposición externa
    d_neut = d_neut_pos + d_neut_neg
    d_ext = np.zeros(n, dtype=float)
    mask_polar = lab != 0
    d_ext[mask_polar] = d_inter[mask_polar] + d_neut[mask_polar]
    # For neutral nodes: exposure to both poles
    d_ext[~mask_polar] = d_S1[~mask_polar] + d_S2[~mask_polar]
    P_ext = np.where(d_total > 0, d_ext / d_total, 0.0)

    return {
        'd_intra_pos': d_intra_pos,
        'd_intra_neg': d_intra_neg,
        'd_inter_pos': d_inter_pos,
        'd_inter_neg': d_inter_neg,
        'd_neut_pos': d_neut_pos,
        'd_neut_neg': d_neut_neg,
        'd_S1': d_S1,
        'd_S2': d_S2,
        'd_total': d_total,
        'S_eq': S_eq,
        'd_anom': d_anom,
        'P_anom': P_anom,
        'd_ext': d_ext,
        'P_ext': P_ext,
    }


def compute_polar_eig(A_s: csr_matrix, labels: np.ndarray) -> np.ndarray:
    n = A_s.shape[0]
    idx_core = np.where(labels != 0)[0]
    if idx_core.size == 0:
        return np.zeros(n, dtype=float)
    # Build A^(p)_+ on core
    rows, cols = A_s.nonzero()
    mask_core = np.isin(rows, idx_core) & np.isin(cols, idx_core)
    rows_c = rows[mask_core]
    cols_c = cols[mask_core]
    vals = A_s.data[mask_core] * labels[rows_c] * labels[cols_c]
    vals_pos = np.where(vals > 0, vals, 0.0)
    # map rows/cols to compact indices
    pos_map = {v: i for i, v in enumerate(idx_core)}
    r_m = np.vectorize(pos_map.get)(rows_c)
    c_m = np.vectorize(pos_map.get)(cols_c)
    A_p_pos = csr_matrix((vals_pos, (r_m, c_m)), shape=(idx_core.size, idx_core.size))
    x = power_iteration(A_p_pos, iters=40)
    ceig = np.zeros(n, dtype=float)
    ceig[idx_core] = np.abs(x)
    return ceig


def compute_external_brokerage(A_abs: csr_matrix, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = A_abs.shape[0]
    W = np.asarray(A_abs.sum(axis=1)).ravel().astype(float)
    A = A_abs.tocsr()
    bext = np.zeros(n, dtype=float)
    C_ext = np.zeros(n, dtype=float)
    for u in range(n):
        cu = labels[u]
        if cu == 0 or W[u] <= 0:
            bext[u] = 0.0
            C_ext[u] = 0.0
            continue
        ru = slice(A.indptr[u], A.indptr[u+1])
        nbrs = A.indices[ru]
        w_uv = A.data[ru]
        # External neighbors (opposite block)
        ext_idx_mask = (labels[nbrs] == -cu)
        extN = nbrs[ext_idx_mask]
        extW = w_uv[ext_idx_mask]
        k = extN.size
        if k == 0:
            bext[u] = 0.0
            C_ext[u] = 0.0
            continue
        p_uv = extW / W[u]
        # External clustering among extN
        # Count edges among extN
        edges_in_ext = 0.0; possible = k * (k - 1) / 2.0
        # For speed, build set for membership and iterate neighbors' neighbors
        ext_set = set(extN.tolist())
        for v in extN:
            rv = slice(A.indptr[v], A.indptr[v+1])
            v_n = A.indices[rv]
            # count only w>v to avoid double
            edges_in_ext += np.count_nonzero(np.isin(v_n, list(ext_set)))
        # Each undirected edge counted twice (for v and w), adjust
        edges_in_ext = edges_in_ext / 2.0
        C_ext[u] = (edges_in_ext / possible) if possible > 0 else 0.0
        # m_bar between ext neighbors (row-stochastic per v over ext neighborhood)
        # Precompute row sums for each v over ext set
        m_rowsums = {}
        for v in extN:
            rv = slice(A.indptr[v], A.indptr[v+1])
            v_n = A.indices[rv]
            v_w = A.data[rv]
            mask_ext = np.isin(v_n, extN)
            denom_v = float(np.sum(v_w[mask_ext]))
            m_rowsums[v] = denom_v
        # Compute brokerage
        # Map extN to positions for p_uw lookup
        idx_map = {node: i for i, node in enumerate(extN)}
        for i_v, v in enumerate(extN):
            denom_v = m_rowsums[v]
            if denom_v <= 0:
                inner = 0.0
            else:
                rv = slice(A.indptr[v], A.indptr[v+1])
                v_n = A.indices[rv]
                v_w = A.data[rv]
                mask_ext = np.isin(v_n, extN)
                neigh_w = v_w[mask_ext]
                neigh_nodes = v_n[mask_ext]
                # sum_w p_uw * m_vw
                inner = 0.0
                for node_w, w_val in zip(neigh_nodes, neigh_w):
                    j = idx_map.get(node_w)
                    if j is None:
                        continue
                    inner += p_uv[j] * (w_val / denom_v)
            bext[u] += p_uv[i_v] * (1.0 - inner)
    return bext, C_ext


def compute_indf_soft(A_s: csr_matrix, labels: np.ndarray, tau: float = 1.0) -> np.ndarray:
    n = A_s.shape[0]
    A_norm = symmetric_normalize(A_s.copy())
    c_vec = labels.astype(float)  # -1,0,1
    # s_plus = sum_v tildeA_uv * c(v)
    s_plus = A_norm @ c_vec
    # Two-class softmax over {+1,-1}
    z1 = tau * s_plus
    z2 = -z1
    # avoid overflow
    m = np.maximum(z1, z2)
    e1 = np.exp(z1 - m)
    e2 = np.exp(z2 - m)
    q1 = e1 / (e1 + e2 + 1e-12)
    q2 = 1.0 - q1
    qmax = np.maximum(q1, q2)
    indf = 2.0 * (1.0 - qmax)
    return np.clip(indf, 0.0, 1.0)


def compute_U_scores(A_s: csr_matrix, labels: np.ndarray, omega: float = 0.65,
                     beta_sig: float = 1.0, eta_bias: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute U_inter, U_intra, and U = omega*U_inter + (1-omega)*U_intra using candidates at 2 hops.
    Follows definitions with Adamic-Adar corrected existence probability and sign probability from triads + prior.
    """
    n = A_s.shape[0]
    A_abs = csr_matrix((np.abs(A_s.data), A_s.indices, A_s.indptr), shape=A_s.shape)
    A_norm = symmetric_normalize(A_s.copy())
    # Build adjacency lists with weights for normalized A
    A_norm = A_norm.tocsr()
    norm_rows = [dict(zip(A_norm.indices[A_norm.indptr[i]:A_norm.indptr[i+1]],
                          A_norm.data[A_norm.indptr[i]:A_norm.indptr[i+1]])) for i in range(n)]
    # Unsigned neighbors for Adamic-Adar
    A_abs = A_abs.tocsr()
    abs_sets = [set(A_abs.indices[A_abs.indptr[i]:A_abs.indptr[i+1]]) for i in range(n)]
    deg_abs = np.asarray(A_abs.sum(axis=1)).ravel().astype(float)
    logdeg = np.log1p(deg_abs)
    inv_logdeg = np.where(logdeg > 0, 1.0 / logdeg, 0.0)
    E_log = np.mean(logdeg[logdeg > 0]) if np.any(logdeg > 0) else 1.0

    U_inter = np.zeros(n, dtype=float)
    U_intra = np.zeros(n, dtype=float)

    for u in range(n):
        cu = labels[u]
        # 2-hop candidates: union of neighbors of neighbors
        two_hop = set()
        for w in abs_sets[u]:
            two_hop.update(abs_sets[w])
        if u in two_hop:
            two_hop.remove(u)
        # exclude direct neighbors (only non-edges)
        two_hop = two_hop.difference(abs_sets[u])
        if not two_hop:
            continue
        # Split by intra/inter wrt u's block; if u neutral, treat all as inter
        if cu == 0:
            inter_cands = list(two_hop)
            intra_cands = []
        else:
            inter_cands = [v for v in two_hop if labels[v] == -cu]
            intra_cands = [v for v in two_hop if labels[v] == cu]

        def compute_channel(cands: list) -> float:
            if not cands:
                return 0.0
            e_vals = []
            var_vals = []
            # Precompute for u
            Nu = abs_sets[u]
            Nu_norm = norm_rows[u]
            for v in cands:
                # Adamic-Adar corrected
                Nv = abs_sets[v]
                common = Nu.intersection(Nv)
                if common:
                    phi = np.sum(inv_logdeg[list(common)])
                else:
                    phi = 0.0
                psi = (logdeg[u] * logdeg[v]) / (E_log * E_log) if E_log > 0 else 1.0
                e_raw = phi / (psi + 1e-12)
                e_vals.append(e_raw)
                # Sign probability via triads + prior
                # s_uv = sum_w tildeA_uw * tildeA_wv = inner product over common normalized neighbors
                s_uv = 0.0
                Nv_norm = norm_rows[v]
                for w in common:
                    s_uv += Nu_norm.get(w, 0.0) * Nv_norm.get(w, 0.0)
                same_block = (labels[v] == cu and cu != 0)
                bias = (eta_bias / (logdeg[u] * logdeg[v] + 1e-12)) if (logdeg[u] > 0 and logdeg[v] > 0) else 0.0
                b_uv = bias if same_block else -bias
                p_pos = 1.0 / (1.0 + np.exp(-beta_sig * (s_uv + b_uv)))
                var_vals.append(4.0 * p_pos * (1.0 - p_pos))
            # Normalize e_vals to [0,1] per node-channel
            e_vals = np.asarray(e_vals, dtype=float)
            if np.allclose(e_vals, e_vals[0]) and e_vals.size > 0:
                e_norm = np.ones_like(e_vals)
            else:
                mn = np.min(e_vals); mx = np.max(e_vals)
                e_norm = (e_vals - mn) / (mx - mn + 1e-12)
            var_vals = np.asarray(var_vals, dtype=float)
            num = float(np.sum(e_norm * var_vals))
            den = float(np.sum(e_norm) + 1e-12)
            return num / den

        U_inter[u] = compute_channel(inter_cands)
        U_intra[u] = compute_channel(intra_cands)

    U = float(omega) * U_inter + (1.0 - float(omega)) * U_intra
    return U_inter, U_intra, U


def run_end_to_end(
    df: pd.DataFrame,
    alpha: float = 1.0,
    beta: float = 5e-3,
    alpha0: float = 1.0,
    tau_indf: float = 1.0,
    omega: float = 0.65,
    beta_sig: float = 1.0,
    eta_bias: float = 0.5,
    gamma: float = 0.5,
    minimal: bool = False,
) -> Dict[str, object]:
    # Build signed adjacency
    A_s, node_to_idx, idx_to_node, n_nodes, _ = preprocess_and_create_signed_adjacency_matrix(
        df_input=df,
        from_node_col='FROM_NODE',
        to_node_col='TO_NODE',
        sign_col='SIGN',
        node_norm_strategy='lower_unidecode_strip',
        weighting_strategy='binary_sum_signs_actual',
    )
    if n_nodes == 0 or A_s.nnz == 0:
        raise ValueError('Grafo vacío o sin aristas válidas')

    # PCD local search labels {-1,0,1}
    y_pcd, pol = run_local_search_paper_k2(A_s, n_nodes, ls_alpha=float(alpha), ls_beta=float(beta), ls_max_iter=2000)
    y = y_pcd.astype(int)

    if minimal:
        df_nodes = pd.DataFrame({
            'node': [idx_to_node[i] for i in range(n_nodes)],
            'cluster': y.astype(int),
        })
        return {
            'A': A_s,
            'labels': y,
            'nodes_df': df_nodes,
            'polarity': pol,
        }

    # Split adjacency
    A_pos, A_neg, A_abs = split_signed(A_s)

    # Partition-based metrics
    part = compute_partition_metrics(A_pos, A_neg, A_abs, y)

    # Polar eig centrality
    ceig = compute_polar_eig(A_s, y)

    # External brokerage + redundancy
    bext, C_ext = compute_external_brokerage(A_abs, y)

    # Importance structural indices
    # For polar nodes, d_intra is degree (pos+neg) to own pole; neutral -> 0
    d_intra = np.zeros(n_nodes, dtype=float)
    mask_S1 = (y == 1); mask_S2 = (y == -1)
    d_intra[mask_S1] = part['d_S1'][mask_S1]
    d_intra[mask_S2] = part['d_S2'][mask_S2]
    # Use definitions for final SI
    SI_deg = normalize_vec(d_intra) * normalize_vec(part['d_ext'] * (1.0 - C_ext))
    delta = 0.6  # Peso para Bext
    SI_esp = (normalize_vec(ceig) ** (1.0 - delta)) * (normalize_vec(bext) ** delta)
    # Heuristic: external entropy + opposition intensity
    w_ext_pos = part['d_inter_pos']
    w_ext_neg = part['d_inter_neg']
    p_plus = (w_ext_pos + alpha0) / (w_ext_pos + w_ext_neg + 2.0 * alpha0)
    H_ext = -(p_plus * np.log(np.clip(p_plus, 1e-12, 1.0)) + (1.0 - p_plus) * np.log(np.clip(1.0 - p_plus, 1e-12, 1.0)))
    pi_opp = np.where((w_ext_pos + w_ext_neg) > 0, w_ext_neg / (w_ext_pos + w_ext_neg), 0.0)

    # Soft frontier index
    indf_soft = compute_indf_soft(A_s, y, tau=float(tau_indf))

    # U scores (possible links uncertainty)
    U_inter, U_intra, U = compute_U_scores(A_s, y, omega=float(omega), beta_sig=float(beta_sig), eta_bias=float(eta_bias))

    # SH* scores (two variants)
    U_hat = normalize_vec(U)
    SIdeg_hat = normalize_vec(SI_deg)
    SIesp_hat = normalize_vec(SI_esp)
    # sqrt form equals gamma=0.5 in power mean
    SH_star_deg = np.sqrt(np.clip(SIdeg_hat, 0, 1) * np.clip(U_hat, 0, 1))
    SH_star_esp = np.sqrt(np.clip(SIesp_hat, 0, 1) * np.clip(U_hat, 0, 1))
    # gamma-weighted generalization
    SH_star_deg_gamma = (np.clip(SIdeg_hat, 0, 1) ** float(gamma)) * (np.clip(U_hat, 0, 1) ** (1.0 - float(gamma)))
    SH_star_esp_gamma = (np.clip(SIesp_hat, 0, 1) ** float(gamma)) * (np.clip(U_hat, 0, 1) ** (1.0 - float(gamma)))

    # Tipología operativa por medianas
    si_med = np.nanmedian(SIesp_hat)
    u_med = np.nanmedian(U_hat)
    def classify(si, u):
        if si >= si_med and u >= u_med:
            return 'sh-spanner'
        if si >= si_med and u < u_med:
            return 'nucleo-duro'
        if si < si_med and u >= u_med:
            return 'pivote-volatil'
        return 'periferico-leal'
    typology = [classify(s, u) for s, u in zip(SIesp_hat, U_hat)]

    # Build DataFrame
    df_nodes = pd.DataFrame({
        'node': [idx_to_node[i] for i in range(n_nodes)],
        'cluster': y.astype(int),
        'deg_total': part['d_total'],
        'S_eq': part['S_eq'],
        'd_anom': part['d_anom'],
        'P_anom': part['P_anom'],
        'd_ext': part['d_ext'],
        'P_ext': part['P_ext'],
        'C_eig': ceig,
        'Bext': bext,
        'C_ext': C_ext,
        'SI_deg': SI_deg,
        'SI_esp': SI_esp,
        'H_ext': H_ext,
        'pi_opp': pi_opp,
        'IndF_soft': indf_soft,
        'U_inter': U_inter,
        'U_intra': U_intra,
        'U': U,
        'SIdeg_hat': SIdeg_hat,
        'SIesp_hat': SIesp_hat,
        'U_hat': U_hat,
        'SH_star_deg': SH_star_deg,
        'SH_star_esp': SH_star_esp,
        'SH_star_deg_gamma': SH_star_deg_gamma,
        'SH_star_esp_gamma': SH_star_esp_gamma,
        'typology': typology,
    })

    return {
        'A': A_s,
        'labels': y,
        'nodes_df': df_nodes,
        'polarity': pol,
    }


def parse_args():
    p = argparse.ArgumentParser(description='PCD Manual E2E: Local Search + métricas avanzadas (sin NN)')
    p.add_argument('--input', nargs='+', required=True, help='CSV(s) con FROM_NODE, TO_NODE, SIGN')
    p.add_argument('--results-dir', default=os.path.join(PROJECT_ROOT, 'results'), help='Carpeta resultados')
    p.add_argument('--name', help='Nombre base del dataset (si es un único archivo)')
    p.add_argument('--topk', type=int, default=20, help='Top-k por SH* (esp)')
    # PCD
    p.add_argument('--alpha', type=float, default=1.0)
    p.add_argument('--beta', type=float, default=5e-3)
    # Heurísticos/prob
    p.add_argument('--alpha0', type=float, default=1.0, help='Suavizado Beta en H_ext')
    p.add_argument('--tau-indf', type=float, default=1.0, help='Temperatura para IndF_soft')
    p.add_argument('--omega', type=float, default=0.65, help='Peso U_inter en U')
    p.add_argument('--beta-sig', type=float, default=1.0, help='Escala en p^{(+)} triádico')
    p.add_argument('--eta-bias', type=float, default=0.5, help='Peso del prior log-degree en p^{(+)}')
    p.add_argument('--gamma', type=float, default=0.5, help='Exponente en SH*_gamma')
    p.add_argument('--minimal', action='store_true', help='Devuelve solo asignación de clúster por nodo')
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(42)  # Puedes usar cualquier número entero, 42 es una convención.

    # Expand inputs
    inputs = []
    for pattern in args.input:
        expanded = glob.glob(pattern)
        if expanded:
            inputs.extend(expanded)
        else:
            print(f'[WARN] No se encontraron archivos para: {pattern}')
    if not inputs:
        print('No hay entradas. Usa --input "data/raw/*.csv"')
        sys.exit(1)

    os.makedirs(args.results_dir, exist_ok=True)

    for ipath in inputs:
        dataset_name = args.name if (args.name and len(inputs) == 1) else os.path.splitext(os.path.basename(ipath))[0]
        out_dir = os.path.join(args.results_dir, f"{dataset_name}_pcd_manual")
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n=== [PCD-Manual] Dataset: {dataset_name} ===")

        df_raw = pd.read_csv(ipath)
        res = run_end_to_end(
            df=df_raw,
            alpha=float(args.alpha), beta=float(args.beta),
            alpha0=float(args.alpha0), tau_indf=float(args.tau_indf),
            omega=float(args.omega), beta_sig=float(args.beta_sig), eta_bias=float(args.eta_bias),
            gamma=float(args.gamma),
            minimal=bool(args.minimal),
        )

        df_nodes = res['nodes_df']  # type: ignore
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')

        if args.minimal:
            nodes_path = os.path.join(out_dir, f'clusters_{ts}.csv')
            df_nodes.to_csv(nodes_path, index=False, encoding='utf-8-sig')
            print(f"Guardado: {nodes_path}")
            continue

        # Choose ranking column: use gamma variant if gamma != 0.5
        try:
            gval = float(args.gamma)
        except Exception:
            gval = 0.5
        rank_col = 'SH_star_esp_gamma' if abs(gval - 0.5) > 1e-12 else 'SH_star_esp'

        # Save CSVs
        nodes_path = os.path.join(out_dir, f'nodes_metrics_pcd_manual_{ts}.csv')
        df_nodes.to_csv(nodes_path, index=False, encoding='utf-8-sig')

        # Top-k by selected SH* (esp) variant
        df_rank = df_nodes.copy()
        df_rank = df_rank.sort_values(rank_col, ascending=False)
        topk = int(max(1, args.topk))
        # Always include both SH* variants for clarity
        cols = ['node', 'cluster', rank_col, 'SH_star_esp', 'SH_star_esp_gamma', 'SH_star_deg', 'SI_esp', 'Bext', 'C_eig', 'U', 'typology']
        # Filter to existing columns (forward-compatible)
        cols = [c for c in cols if c in df_rank.columns]
        topk_df = df_rank.head(topk)[cols]
        topk_path = os.path.join(out_dir, f'top_{topk}_sh_star_esp_{ts}.csv')
        topk_df.to_csv(topk_path, index=False, encoding='utf-8-sig')

        print(f"Guardado: {nodes_path}")
        print(f"Top-{topk} por {rank_col} (gamma={gval}) guardado en: {topk_path}")
        print('Vista previa Top-10:')
        print(topk_df.head(10).round(4).to_string(index=False))


if __name__ == '__main__':
    main()
