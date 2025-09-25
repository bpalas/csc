import pandas as pd
from unidecode import unidecode
from scipy.sparse import csr_matrix
from collections import defaultdict
from pandas.tseries.offsets import DateOffset




def normalizar_nombres_nodos_dispatch(df: pd.DataFrame,
                                     columnas_nodos: list = ['FROM_NODE', 'TO_NODE'],
                                     strategy='lower_unidecode_strip') -> pd.DataFrame:
    df_norm = df.copy()
    for col in columnas_nodos:
        if col in df_norm.columns:
            series = df_norm[col].astype(pd.StringDtype())
            if strategy == 'lower_unidecode_strip':
                series = series.str.lower()
                series = series.apply(lambda x: unidecode(str(x)) if pd.notna(x) and isinstance(x, str) else x)
                series = series.str.strip()
            elif strategy == 'upper_strip':
                series = series.str.upper()
                series = series.str.strip()
            elif strategy == 'raw': pass
            else: raise ValueError(f"Estrategia de normalización de nodos desconocida: {strategy}")
            df_norm[col] = series
        else: print(f"Advertencia: Columna '{col}' no encontrada para normalizar.")
    return df_norm

def preprocess_and_create_signed_adjacency_matrix(
    df_input: pd.DataFrame, from_node_col='FROM_NODE', to_node_col='TO_NODE',
    sign_col='SIGN',
    node_norm_strategy='lower_unidecode_strip',
    weighting_strategy='binary_sum_signs_actual',
    tanh_scale_factor=1.0 # No se usa directamente en las ramas activas de esta versión simplificada
):
    if not isinstance(df_input, pd.DataFrame): raise ValueError("df_input debe ser un DataFrame.")
    required_cols = [from_node_col, to_node_col, sign_col]
    if not all(col in df_input.columns for col in required_cols):
        raise ValueError(f"Faltan columnas: {[c for c in required_cols if c not in df_input.columns]}")

    df = df_input[required_cols].copy()
    # La normalización de nodos se aplica aquí a las columnas especificadas del DataFrame de entrada
    df = normalizar_nombres_nodos_dispatch(df, [from_node_col, to_node_col], strategy=node_norm_strategy)

    sign_map = {'positive': 1, 'negative': -1, 'neutral': 0,
                'positivo': 1, 'negativo': -1, 'neutro':0}
    def map_sign_value(val):
        try: int_val = int(val); return 1 if int_val > 0 else -1 if int_val < 0 else 0
        except (ValueError, TypeError):
            s_val = str(val).lower();
            try: s_val = unidecode(s_val) # Manejar acentos
            except Exception: pass
            return sign_map.get(s_val, 0) # Default a 0 si no está en el mapa
    df['sign_value'] = df[sign_col].apply(map_sign_value).astype(int)

    all_nodes_series = pd.concat([df[from_node_col], df[to_node_col]])
    unique_nodes = sorted(all_nodes_series.astype(str).dropna().unique())
    node_to_idx = {n: i for i, n in enumerate(unique_nodes)}
    idx_to_node = {i: n for i, n in enumerate(unique_nodes)}
    n_nodes = len(unique_nodes)
    # ordered_node_ids no es estrictamente necesario para devolver A_s pero se incluye por consistencia con el original
    ordered_node_ids = [idx_to_node.get(i) for i in range(n_nodes)] if n_nodes > 0 else []


    if n_nodes == 0:
        return csr_matrix((0,0), dtype=float), node_to_idx, idx_to_node, n_nodes, ordered_node_ids

    agg_details = defaultdict(lambda: {'sum_signs_actual': 0.0, 'n_pos': 0, 'n_neg': 0, 'n_neu': 0, 'n_total': 0})
    for _, r in df.iterrows():
        idx1_str, idx2_str = str(r[from_node_col]), str(r[to_node_col])
        if idx1_str not in node_to_idx or idx2_str not in node_to_idx : continue # Nodo no en unique_nodes
        idx1, idx2 = node_to_idx[idx1_str], node_to_idx[idx2_str]
        if idx1 == idx2: continue # Omitir auto-bucles
        key = tuple(sorted((idx1, idx2))) # Clave única para el par de nodos
        details = agg_details[key]
        sv = r['sign_value']
        details['sum_signs_actual'] += float(sv) # Para binary_sum_signs_actual
        details['n_total'] += 1
        if sv == 1: details['n_pos'] += 1
        elif sv == -1: details['n_neg'] += 1
        else: details['n_neu'] += 1 # Contar neutrales si existen y son mapeados a 0

    rows, cols, data_vals = [], [], []
    for (i1, i2), d in agg_details.items():
        w = 0.0
        if weighting_strategy == 'binary_sum_signs_actual':
            if d['sum_signs_actual'] > 0: w = 1.0
            elif d['sum_signs_actual'] < 0: w = -1.0
            # Si sum_signs_actual es 0 (ej. +1 y -1), w permanece 0 (neutral)
        elif weighting_strategy == 'sum_raw': # n_pos - n_neg
             w = float(d['n_pos'] - d['n_neg'])
        # Añadir otras estrategias de ponderación aquí si se necesitan, como 'tanh_sum', etc.
        else:
            raise ValueError(f"Estrategia de ponderación desconocida: {weighting_strategy}")
        
        if abs(w) > 1e-9: # Solo añadir si no es cero (o muy cercano a cero)
            rows.extend([i1, i2]); cols.extend([i2, i1]); data_vals.extend([w, w])

    A_s_matrix = csr_matrix((data_vals, (rows, cols)), shape=(n_nodes, n_nodes), dtype=float)
    A_s_matrix.sum_duplicates() # Sumar pesos si hay múltiples aristas (ahora agregadas) entre los mismos nodos
    A_s_matrix.eliminate_zeros() # Eliminar ceros explícitos
    return A_s_matrix, node_to_idx, idx_to_node, n_nodes, ordered_node_ids
import pandas as pd
import datetime
from pandas.tseries.offsets import DateOffset

def cargar_datos_2020(ruta_archivo, columna_fecha='date'):

    df = pd.read_csv(ruta_archivo)
    
    df[columna_fecha] = pd.to_datetime(df[columna_fecha]).dt.tz_localize(None)
    

    
    return df


# --- FUNCIÓN PARA CARGAR Y DIVIDIR DATOS DEL PLEBISCITO 2022 ---
def cargar_datos_2022_pre_post(ruta_archivo):
    """
    Carga los datos del plebiscito de 2022 y los divide en dos DataFrames: pre y post.
    - Aplica la limpieza de texto en las columnas de nodos.
    - Filtra 3 meses antes y 3 meses después del evento.
    - Excluye explícitamente el día del plebiscito (4 de septiembre de 2022).
    """
    print("Cargando y dividiendo datos para df_2022_pre y df_2022_post...")
    df = pd.read_csv(ruta_archivo)

    # Limpieza de texto (nodos sin números y con más de una palabra)
    df = df[~df['FROM_NODE'].str.contains(r'\d', na=False) & ~df['TO_NODE'].str.contains(r'\d', na=False)]
    df = df[df['FROM_NODE'].str.split().str.len() > 1]
    df = df[df['TO_NODE'].str.split().str.len() > 1]

    # Asegurar que la columna de fecha sea de tipo datetime
    df['DATE'] = pd.to_datetime(df['DATE']).dt.tz_localize(None)
    
    # Fecha del evento de 2022
    fecha_evento_2022 = pd.Timestamp('2022-09-04')
    
    # Definir el rango total de 6 meses
    fecha_inicio = fecha_evento_2022 - DateOffset(months=3)
    fecha_fin = fecha_evento_2022 + DateOffset(months=3)
    
    # Filtrar primero por el rango general
    df_rango_total = df[(df['DATE'] >= fecha_inicio) & (df['DATE'] <= fecha_fin)]

    # --- División en PRE y POST, excluyendo el día del evento ---
    df_pre = df_rango_total[df_rango_total['DATE'] < fecha_evento_2022].copy()
    df_post = df_rango_total[df_rango_total['DATE'] > fecha_evento_2022].copy()
    
    print(f"-> Se cargaron {len(df_pre)} filas en df_2022_pre.")
    print(f"-> Se cargaron {len(df_post)} filas en df_2022_post.\n")
    
    return df_pre, df_post