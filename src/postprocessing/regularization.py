import gc
import numpy as np
import pandas as pd
from scipy.linalg import solve_banded
from tqdm import tqdm


def calculate_ensemble_confidence(preds_matrix, lower_ci_pct=10.0, upper_ci_pct=90.0, gamma=3.0):
    """Computes ensemble percentiles, CV, and ML Confidence Score."""
    median_pred = np.percentile(preds_matrix, 50.0, axis=1)
    lower_ci = np.percentile(preds_matrix, lower_ci_pct, axis=1)
    upper_ci = np.percentile(preds_matrix, upper_ci_pct, axis=1)
    
    std_pred = np.std(preds_matrix, axis=1)
    mean_pred = np.mean(preds_matrix, axis=1)
    
    cv = std_pred / (mean_pred + 1e-8)
    confidence_score = np.exp(-gamma * cv) * 100.0
    
    return {
        'median': median_pred,
        'lower_ci': lower_ci,
        'upper_ci': upper_ci,
        'std': std_pred,
        'cv': cv,
        'confidence_score': confidence_score
    }


def topological_sort_mainstem(grp, comid_col_name='flowpath_id', toid_col_name='flowpath_toid'):
    """Performs topological sort along a single mainstem level path."""
    ids = set(grp[comid_col_name].values)
    to_map = dict(zip(grp[comid_col_name], grp[toid_col_name]))
    in_degree = {i: 0 for i in ids}
    
    for i in ids:
        to_id = to_map.get(i)
        if to_id in in_degree:
            in_degree[to_id] += 1
            
    queue = [i for i in ids if in_degree[i] == 0]
    sorted_ids = []
    
    while queue:
        curr = queue.pop(0)
        sorted_ids.append(curr)
        to_id = to_map.get(curr)
        if to_id in in_degree:
            in_degree[to_id] -= 1
            if in_degree[to_id] == 0:
                queue.append(to_id)
                
    remaining = [i for i in ids if i not in sorted_ids]
    return sorted_ids + remaining


def regularize_mainstem_roughness_ensemble(
    flowlines_df,
    ensemble_preds_matrix,
    lambda_single=10.0,
    gamma_reg=0.01,
    fallback_n_single=0.035,
    comid_col_name='flowpath_id',
    mainstem_col_name='mainstemlp',
    toid_col_name='flowpath_toid'
):
    """
    Applies Gaussian Markov Random Field (GMRF) topological regularization along mainstems
    using a tridiagonal banded matrix solver.
    """
    num_models = ensemble_preds_matrix.shape[1]
    pred_cols = [f"pred_m_{i}" for i in range(num_models)]
    
    preds_df = pd.DataFrame(ensemble_preds_matrix, columns=pred_cols, index=flowlines_df.index)
    work_df = pd.concat([flowlines_df[[comid_col_name, mainstem_col_name, toid_col_name]], preds_df], axis=1)
    
    grouped = work_df.groupby(mainstem_col_name, sort=False)
    regularized_arrays = []
    all_sorted_ids = []
    
    for mainstem_id, grp in tqdm(grouped, desc="Multi-Column GMRF Mainstem Regularization"):
        N = len(grp)
        if N == 0:
            continue
            
        sorted_ids = topological_sort_mainstem(grp, comid_col_name=comid_col_name, toid_col_name=toid_col_name)
        grp_sorted = grp.set_index(comid_col_name).reindex(sorted_ids)
        
        Y_mainstem = grp_sorted[pred_cols].values
        Y_mainstem = np.nan_to_num(Y_mainstem, nan=fallback_n_single)
        
        if N == 1:
            regularized_arrays.append(Y_mainstem)
            all_sorted_ids.extend(sorted_ids)
            continue
            
        main_diag = np.full(N, 1.0 + 2.0 * lambda_single + gamma_reg, dtype=np.float32)
        main_diag[0] -= lambda_single
        main_diag[-1] -= lambda_single
        
        off_diag = np.full(N - 1, -lambda_single, dtype=np.float32)
        
        ab = np.zeros((3, N), dtype=np.float32)
        ab[0, 1:] = off_diag
        ab[1, :] = main_diag
        ab[2, :-1] = off_diag
        
        W_diag = 1.0
        B = np.asfortranarray(W_diag * Y_mainstem + gamma_reg * fallback_n_single, dtype=np.float32)
        X_reg = solve_banded((1, 1), ab, B)
        
        regularized_arrays.append(X_reg)
        all_sorted_ids.extend(sorted_ids)

    del work_df, preds_df
    gc.collect()

    stacked_regularized = np.vstack(regularized_arrays)
    regularized_results_df = pd.DataFrame(stacked_regularized, columns=pred_cols)
    regularized_results_df[comid_col_name] = all_sorted_ids

    meta_cols = [c for c in ['lon', 'lat', 'streamorder', 'HLR', 'mainstemlp', 'flowpath_toid'] if c in flowlines_df.columns and c not in regularized_results_df.columns]
    if meta_cols:
        regularized_results_df = regularized_results_df.merge(flowlines_df[[comid_col_name] + meta_cols], on=comid_col_name, how='left')

    return regularized_results_df