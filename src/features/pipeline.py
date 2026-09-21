import numpy as np
import pandas as pd
import shapely


def engineer_flowline_features(
    df: pd.DataFrame, 
    target_stage: str = "tw_bf",
    length_threshold_km: float = 0.1,
    epsilon: float = 1e-6,
    min_slope: float = 1e-4,
    min_totda: float = 1e-3,
    max_sinuosity: float = 5.0
) -> pd.DataFrame:
    """
    Vectorized and highly optimized feature engineering pipeline for NHD flowline networks.
    Designed to process 2.7+ million rows efficiently.
    """
    print(f"Initializing copy of flowline network for stage: {target_stage}...")
    df = df.copy()
    
    assert hasattr(shapely, "get_point"), "Shapely 2.0 or newer is required for this pipeline."
    
    if target_stage in ["y_bf", "r", "n"]:
        assert "tw_bf_pred" in df.columns, "Stage 2+ feature engineering requires the predicted top width column 'tw_bf_pred'."
    if target_stage in ["r", "n"]:
        assert "y_bf_pred" in df.columns, "Stage 3+ feature engineering requires the predicted bankfull depth column 'y_bf_pred'."
    if target_stage == "n":
        assert "r_bf_pred" in df.columns, "Stage 4 feature engineering requires the predicted Dingman r shape exponent column 'r_bf_pred'."

    print("Extracting vectorized geometry coordinates...")
    geoms = df['geometry'].values
    
    num_parts = shapely.get_num_geometries(geoms)
    first_parts = shapely.get_geometry(geoms, 0)
    last_parts = shapely.get_geometry(geoms, num_parts - 1)
    start_pts = shapely.get_point(first_parts, 0)
    end_pts = shapely.get_point(last_parts, -1)
    curv_lengths = shapely.length(geoms)
    euclidean_dists = shapely.distance(start_pts, end_pts)
    
    sinuosity_default = np.maximum(
        np.where(euclidean_dists > 0, curv_lengths / euclidean_dists, 1.0), 
        1.0
    )
    
    print("Mapping DAG network topology and Mainstem Level Paths...")
    up_candidates = df[['flowpath_id', 'flowpath_toid', 'mainstemlp', 'totdasqkm']].dropna(subset=['flowpath_toid']).rename(
        columns={
            'flowpath_id': 'flowpath_id_up',
            'mainstemlp': 'mainstemlp_up'
        }
    )
    
    down_targets = df[['flowpath_id', 'mainstemlp']].rename(
        columns={
            'flowpath_id': 'flowpath_id_down',
            'mainstemlp': 'mainstemlp_down'
        }
    )
    
    candidates_merged = up_candidates.merge(
        down_targets,
        left_on='flowpath_toid',
        right_on='flowpath_id_down',
        how='inner'
    )
    
    candidates_merged['same_mainstem'] = (candidates_merged['mainstemlp_up'] == candidates_merged['mainstemlp_down']).astype(int)
    
    candidates_sorted = candidates_merged.sort_values(
        by=['flowpath_toid', 'same_mainstem', 'totdasqkm'],
        ascending=[True, False, False]
    )
    
    dominant_upstream_match = candidates_sorted.drop_duplicates(subset=['flowpath_toid'])
    
    unique_id_mask = ~df['flowpath_id'].duplicated()
    id_to_index = pd.Series(df.index[unique_id_mask], index=df['flowpath_id'][unique_id_mask])
    dominant_upstream_match['upstream_row_idx'] = dominant_upstream_match['flowpath_id_up'].map(id_to_index)
    
    idx_map = pd.Series(dominant_upstream_match['upstream_row_idx'].values, index=dominant_upstream_match['flowpath_toid'])
    
    upstream_idx_col = df['flowpath_id'].map(idx_map)
    upstream_idx_arr = upstream_idx_col.fillna(-1).astype(int).values
    
    if 'lengthkm' in df.columns:
        is_short = df['lengthkm'].values <= length_threshold_km
    else:
        is_short = (curv_lengths / 1000.0) <= length_threshold_km
        
    to_merge = is_short & (upstream_idx_arr != -1)
    
    print("Resolving stabilized sinuosity for short flowlines...")
    u_idx = upstream_idx_arr[to_merge]
    
    merged_length = curv_lengths[to_merge] + curv_lengths[u_idx]
    merged_dist = shapely.distance(start_pts[u_idx], end_pts[to_merge])
    
    merged_sinuosity = np.maximum(
        np.where(merged_dist > 0, merged_length / merged_dist, 1.0), 
        1.0
    )
    
    S_stabilized = sinuosity_default.copy()
    S_stabilized[to_merge] = merged_sinuosity
    S_stabilized = np.clip(S_stabilized, 1.0, max_sinuosity)
    df['S_stabilized'] = S_stabilized
    
    print("Performing vectorized calculations for hydraulic and geomorphic features...")
    
    slope_arr = df['slope'].values
    totda_arr = df['totdasqkm'].values
    arb_arr = df['arb_sum'].values
    path_arr = df['pathlength'].values
    hydroseq_arr = df['hydroseq'].values
    
    slope_clean = np.maximum(slope_arr, min_slope)
    totda_clean = np.maximum(totda_arr, min_totda)
    
    df['SPI_actual'] = totda_arr * slope_clean
    df['V_bf_proxy'] = (np.power(totda_clean, 0.27) * np.sqrt(slope_clean + epsilon)) / S_stabilized
    df['FCD'] = slope_clean / (np.power(totda_clean, -0.45) + epsilon)
    df['FS_proxy'] = (S_stabilized * np.sqrt(totda_clean)) / (slope_clean + epsilon)
    df['HD'] = arb_arr / (np.power(totda_clean, 0.55) + epsilon)
    df['RP'] = path_arr / (path_arr + arb_arr + epsilon)
    
    trib_sum = df.groupby('flowpath_toid')['totdasqkm'].sum()
    sum_trib_totdasqkm = df['flowpath_id'].map(trib_sum).fillna(0).values
    df['UCR'] = np.divide(totda_arr, sum_trib_totdasqkm, out=np.ones_like(totda_arr, dtype=np.float64), where=(sum_trib_totdasqkm > 0))
    
    print("Calculating topological sequence positions...")
    mainstem_stats = df.groupby('mainstemlp')['hydroseq'].agg(['min', 'max'])
    df_merged = df.join(mainstem_stats, on='mainstemlp')
    
    min_hydroseq = df_merged['min'].values
    max_hydroseq = df_merged['max'].values
    
    df['NMP'] = (hydroseq_arr - min_hydroseq) / (max_hydroseq - min_hydroseq + epsilon)

    if target_stage in ["y_bf", "r", "n"]:
        print("Constructing Stage 2 features (Bankfull Depth Stage)...")
        tw_pred = df['tw_bf_pred'].values
        df['WDR'] = tw_pred / (np.sqrt(totda_clean) + epsilon)
        df['WSP'] = tw_pred * slope_clean

    if target_stage in ["r", "n"]:
        print("Constructing Stage 3 features (Dingman Shape Exponent Stage)...")
        y_pred = df['y_bf_pred'].values
        df['AR_channel'] = tw_pred / (y_pred + epsilon)
        df['HSI'] = (tw_pred * y_pred) / (totda_clean + epsilon)
        df['SWR'] = (slope_clean * y_pred) / (tw_pred + epsilon)

    if target_stage == "n":
        print("Constructing Stage 4 features (Manning's Roughness Stage)...")
        r_pred = df['r_bf_pred'].values
        
        A_bf = (r_pred / (r_pred + 1.0)) * tw_pred * y_pred
        P_bf = tw_pred + ((2.0 * r_pred) / (r_pred + 1.0)) * (np.power(y_pred, 2) / (tw_pred + epsilon))
        R_bf_dingman = A_bf / (P_bf + epsilon)
        
        df['R_bf_dingman'] = R_bf_dingman
        df['RRP'] = 1.0 / (R_bf_dingman + epsilon)
        df['MRP'] = np.power(S_stabilized, 2) - 1.0
        df['u_star'] = np.sqrt(9.81 * R_bf_dingman * slope_clean)

    print("Purging local lengthkm and areasqkm scales...")
    cols_to_drop = [col for col in ['lengthkm', 'areasqkm'] if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    print("Processing complete. All features successfully added.")
    return df