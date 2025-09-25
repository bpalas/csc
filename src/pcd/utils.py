import numpy as np
from scipy.sparse import issparse
from scipy.sparse.linalg import eigsh



def _get_principal_eigenvector(A_s_matrix_input, n_nodes_val, eigen_solver, sparse_tol, v0_eigsh, algo_name):
    v1_vector = None; current_A = A_s_matrix_input; is_A_sparse = issparse(current_A)
    try:
        if eigen_solver == 'numpy_robust':
            A_dense = current_A if not is_A_sparse else current_A.toarray()
            if A_dense.dtype != np.float64: A_dense = A_dense.astype(np.float64)
            eigenvalues, eigenvectors = np.linalg.eig(A_dense)
            real_eigenvalues = np.real(eigenvalues)
            if not np.any(np.isfinite(real_eigenvalues)) and len(real_eigenvalues) > 0:
                 raise np.linalg.LinAlgError(f"{algo_name}: Autovalores no finitos.")
            elif len(real_eigenvalues) == 0 and n_nodes_val > 0 : # No eigenvalues but nodes exist
                 print(f"Advertencia ({algo_name}): No se encontraron autovalores para N={n_nodes_val}. Vector v1 será ceros.")
                 v1_vector = np.zeros(n_nodes_val)
                 return v1_vector # Retornar ceros si no hay autovalores
            elif len(real_eigenvalues) == 0 and n_nodes_val == 0: # No eigenvalues y no nodes
                 return np.array([])


            idx_principal = np.nanargmax(real_eigenvalues); v1_vector = np.real(eigenvectors[:, idx_principal])

        elif eigen_solver == 'scipy':
            A_for_eigsh = current_A.astype(np.float64) # eigsh requiere float
            if n_nodes_val == 0: return np.array([])
            if n_nodes_val == 1:
                # Para N=1, el autovector es [1] si A[0,0] !=0, o [0] si A[0,0]==0, el autovalor es A[0,0]
                v1_vector = np.array([1.0]) if float(A_for_eigsh[0,0]) != 0.0 else np.array([0.0])
            elif n_nodes_val > 1:
                # eigsh requiere k < n_components (n_nodes_val)
                k_compute = 1
                # ncv debe ser > k y <= n_nodes_val. Típicamente min(n_nodes_val, max(2*k + 1, 20))
                ncv_param = min(n_nodes_val, max(2 * k_compute + 1, 20))
                if ncv_param <= k_compute : # Ajuste si n_nodes_val es muy pequeño (e.g., N=2, k=1)
                    ncv_param = k_compute + 1 if n_nodes_val > k_compute else n_nodes_val


                v0_p = np.random.rand(n_nodes_val) if v0_eigsh else None
                try:
                    eigenvalues_eigsh, eigenvectors_eigsh = eigsh(A_for_eigsh, k=k_compute, which='LA', tol=sparse_tol, v0=v0_p, ncv=ncv_param)
                    v1_vector = np.real(eigenvectors_eigsh[:, 0])
                except Exception as e_eigsh:
                    print(f"Error con eigsh ({algo_name}, N={n_nodes_val}): {e_eigsh}. Intentando con np.linalg.eig.")
                    # Fallback a numpy_robust si eigsh falla y la matriz no es demasiado grande
                    if not is_A_sparse or (is_A_sparse and n_nodes_val < 2000): # Límite arbitrario para fallback
                         A_dense_fb = current_A.toarray() if is_A_sparse else current_A
                         if A_dense_fb.dtype != np.float64: A_dense_fb = A_dense_fb.astype(np.float64)
                         eigenvalues_fb, eigenvectors_fb = np.linalg.eig(A_dense_fb)
                         real_eigenvalues_fb = np.real(eigenvalues_fb)
                         if not np.any(np.isfinite(real_eigenvalues_fb)) and len(real_eigenvalues_fb) > 0:
                             raise np.linalg.LinAlgError(f"{algo_name} (fallback): Autovalores no finitos.")
                         elif len(real_eigenvalues_fb) == 0:
                              v1_vector = np.zeros(n_nodes_val) # Si no hay autovalores, v1 es ceros
                         else:
                            idx_principal_fb = np.nanargmax(real_eigenvalues_fb)
                            v1_vector = np.real(eigenvectors_fb[:, idx_principal_fb])
                    else:
                        print(f"Fallback a numpy_robust omitido para {algo_name} debido al tamaño de la matriz dispersa ({n_nodes_val}).")
                        return None


        else: raise ValueError(f"Eigen solver desconocido: {eigen_solver}")
    except np.linalg.LinAlgError as e_la: # Captura errores de álgebra lineal específicamente
        print(f"Error de Álgebra Lineal cálculo autovectores ({algo_name}, {eigen_solver}, N={n_nodes_val}): {e_la}. Vector v1 será ceros.")
        v1_vector = np.zeros(n_nodes_val) # Devuelve ceros si hay error LA irrecuperable
    except ValueError as e_val: # Captura otros ValueError (ej. k > N)
        print(f"Error de Valor cálculo autovectores ({algo_name}, {eigen_solver}, N={n_nodes_val}): {e_val}. Vector v1 será ceros.")
        v1_vector = np.zeros(n_nodes_val)
    except Exception as e: # Captura general
        print(f"Error Genérico cálculo autovectores ({algo_name}, {eigen_solver}, N={n_nodes_val}): {e}. Vector v1 será ceros.")
        v1_vector = np.zeros(n_nodes_val)


    if v1_vector is None : # Si, después de todo, v1_vector es None
        print(f"Advertencia Final ({algo_name}): v1_vector no pudo ser calculado. Se devolverá un vector de ceros.")
        v1_vector = np.zeros(n_nodes_val)
    elif len(v1_vector) != n_nodes_val: # Chequeo de dimensión final
        print(f"Error Dimensión ({algo_name}): v1_vector dim incorrecta ({len(v1_vector)} vs N={n_nodes_val}). Se devolverá un vector de ceros.")
        v1_vector = np.zeros(n_nodes_val)


    v1_vector[~np.isfinite(v1_vector)] = 0.0 # Reemplazar NaN/inf con 0 en el vector final
    return v1_vector
