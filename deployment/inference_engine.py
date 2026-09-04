import gc
import os
import time
import fsspec
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.core.io import load_json, load_parquet, load_pickle, save_parquet
from src.models.architectures import (
    CrossFeatureInteraction,
    ModernResTabBlockV4,
    PhysicalResTabNet,
    PiecewiseLinearEncoding,
    RMSNorm,
    SqueezeAndExcitation1D
)
from src.models.wrappers import PyTorchTabularWrapper
from src.postprocessing.regularization import calculate_ensemble_confidence, regularize_mainstem_roughness_ensemble


def _join_uri(base_path: str, *sub_paths: str) -> str:
    """Joins URI components supporting POSIX/Windows paths and s3:// cloud URIs."""
    if base_path.startswith("s3://"):
        clean_base = base_path.rstrip("/")
        clean_subs = [p.strip("/") for p in sub_paths if p]
        return "/".join([clean_base] + clean_subs)
    return os.path.normpath(os.path.join(base_path, *sub_paths))


def _get_parent_uri(uri: str) -> str:
    """Gets the parent directory URI for local paths and s3:// cloud URIs."""
    clean = uri.rstrip("/\\")
    if clean.startswith("s3://"):
        parts = clean.split("/")
        return "/".join(parts[:-1]) if len(parts) > 3 else clean
    return os.path.dirname(clean)


def _apply_feature_transformations(
    df: pd.DataFrame, 
    feature_columns: list, 
    transform_map: dict, 
    quantile_bounds: dict = None, 
    epsilon: float = 1e-6
) -> tuple:
    """Applies vectorized feature transformations and quantile clipping."""
    out_df = df.copy()
    computed_bounds = {} if quantile_bounds is None else dict(quantile_bounds)

    for col in feature_columns:
        if col in out_df.columns:
            if quantile_bounds is not None and col in quantile_bounds:
                q_low = quantile_bounds[col]["q_low"]
                q_high = quantile_bounds[col]["q_high"]
                out_df[col] = np.clip(out_df[col].values, q_low, q_high)
            elif quantile_bounds is None:
                q_low = float(np.nanpercentile(out_df[col].values, 0.01))
                q_high = float(np.nanpercentile(out_df[col].values, 99.999))
                computed_bounds[col] = {"q_low": q_low, "q_high": q_high}
                out_df[col] = np.clip(out_df[col].values, q_low, q_high)

            if col in transform_map:
                method = transform_map[col]
                if method == "log1p":
                    out_df[col] = np.log1p(np.maximum(0, out_df[col].values))
                elif method == "log":
                    out_df[col] = np.log(np.maximum(out_df[col].values, epsilon))
                elif method == "log10":
                    out_df[col] = np.log10(np.maximum(out_df[col].values, epsilon))

    return out_df, computed_bounds


def predict_model_fast(model_obj, X_mat, batch_size=32768, device=None):
    """Fast mini-batched inference wrapper preventing RAM/VRAM memory spikes."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    if isinstance(model_obj, PyTorchTabularWrapper):
        return model_obj.predict(X_mat, batch_size=batch_size).astype(np.float32)
    elif isinstance(model_obj, torch.nn.Module):
        model_obj.to(device)
        model_obj.eval()
        preds = np.zeros(len(X_mat), dtype=np.float32)
        with torch.inference_mode():
            for start_idx in range(0, len(X_mat), batch_size):
                end_idx = start_idx + batch_size
                batch_tensor = torch.tensor(X_mat[start_idx:end_idx], dtype=torch.float32, device=device)
                out = model_obj(batch_tensor)
                preds[start_idx:end_idx] = out.cpu().numpy().squeeze()
        return preds
    elif hasattr(model_obj, "predict"):
        num_samples = len(X_mat)
        preds = np.zeros(num_samples, dtype=np.float32)
        chunk_sz = 100000
        for start_idx in range(0, num_samples, chunk_sz):
            end_idx = min(start_idx + chunk_sz, num_samples)
            chunk_x = X_mat[start_idx:end_idx]
            preds[start_idx:end_idx] = model_obj.predict(chunk_x).astype(np.float32).ravel()
        return preds
    else:
        raise TypeError("Unsupported model object for inference.")


def run_stage_inference(
    target_var: str,
    input_df: pd.DataFrame,
    model_dir: str,
    model_filename: str = None,
    feature_order_path: str = None,
    imputation_values_path: str = None,
    comid_col_name: str = "flowpath_id",
    mainstem_col_name: str = "mainstemlp",
    toid_col_name: str = "flowpath_toid",
    chunk_size: int = 100000,
    transform_target: str = "log1p",
    is_roughness: bool = False,
    lambda_single: float = 10.0,
    gamma_reg: float = 0.01
) -> pd.DataFrame:
    """
    Executes inference for a target stage using hierarchical bidirectional artifact resolution.
    Searches model_dir, subdirectories, and parent directories (..) automatically.
    """
    print("\n" + "=" * 100)
    print(f"EXECUTING INFERENCE ENGINE FOR TARGET: '{target_var.upper()}'")
    print(f"Model Path Specified: '{model_dir}'")
    print("=" * 100)

    fs_model, _ = fsspec.core.url_to_fs(model_dir)
    if not fs_model.exists(model_dir):
        raise FileNotFoundError(f"CRITICAL: Model directory does not exist: '{model_dir}'")

    clean_dir = model_dir.rstrip("/\\")
    parent_dir = _get_parent_uri(clean_dir)

    target_aliases = [target_var]
    if is_roughness or target_var.startswith("n"):
        for alias in ["n", "n_single", "n_in_channel", "n_out_channel"]:
            if alias not in target_aliases:
                target_aliases.append(alias)

    # Hierarchical Search Base Directories: current folder, subfolders, and parent directory above
    search_base_dirs = [
        clean_dir,
        _join_uri(clean_dir, "n_out"),
        _join_uri(clean_dir, f"{target_var}_out"),
        parent_dir,
        _join_uri(parent_dir, "models"),
        _join_uri(parent_dir, "n_out"),
        _join_uri(parent_dir, f"{target_var}_out")
    ]
    # Deduplicate while preserving order
    search_base_dirs = list(dict.fromkeys(search_base_dirs))

    # Discover Metadata JSON
    meta_candidates = []
    for s_base in search_base_dirs:
        for t_alias in target_aliases:
            meta_candidates.extend([
                _join_uri(s_base, f"transformation_metadata_{t_alias}.json"),
                _join_uri(s_base, f"ensemble_metadata_{t_alias}.json"),
                _join_uri(s_base, "ensemble", f"ensemble_metadata_{t_alias}.json"),
                _join_uri(s_base, f"metadata_{t_alias}.json")
            ])

    transform_map = {}
    quantile_bounds = {}
    imputation_values = {}
    log_mse_val = 0.0
    model_val_mses = {}
    target_min_clip = 0.01 if is_roughness else 0.0
    target_max_clip = 0.35 if is_roughness else float("inf")
    lower_ci_pct = 10.0
    upper_ci_pct = 90.0
    retained_features = None
    meta_loaded_path = None

    for m_cand in meta_candidates:
        if fs_model.exists(m_cand):
            try:
                meta_data = load_json(m_cand)
                meta_loaded_path = m_cand
                print(f"Loaded metadata artifact from: '{m_cand}'")
                
                transform_map = meta_data.get("transform_map", {})
                quantile_bounds = meta_data.get("quantile_bounds", {})
                imputation_values = meta_data.get("imputation_medians", {})
                log_mse_val = meta_data.get("log_mse_val", 0.0)
                model_val_mses = meta_data.get("model_val_mses", {})
                target_min_clip = meta_data.get("target_min_clip", target_min_clip)
                target_max_clip = meta_data.get("target_max_clip", target_max_clip)
                lower_ci_pct = meta_data.get("lower_ci_pct", lower_ci_pct)
                upper_ci_pct = meta_data.get("upper_ci_pct", upper_ci_pct)
                if "retained_features" in meta_data:
                    retained_features = meta_data["retained_features"]
                break
            except Exception as e:
                print(f"Warning: Found metadata candidate '{m_cand}' but failed to parse: {e}")

    # Discover Imputation Values & Feature List (Checks current & parent above)
    if not imputation_values:
        imput_candidates = [imputation_values_path] if imputation_values_path else []
        for s_base in search_base_dirs:
            for t_alias in target_aliases:
                imput_candidates.extend([
                    _join_uri(s_base, "metrics", f"median_imput_{t_alias}.parquet"),
                    _join_uri(s_base, f"median_imput_{t_alias}.parquet")
                ])
        for imp_cand in imput_candidates:
            if imp_cand and fs_model.exists(imp_cand):
                try:
                    median_df = load_parquet(imp_cand)
                    if "Feature" in median_df.columns:
                        imputation_values = median_df.set_index("Feature")["Median"].to_dict()
                    elif "Median" in median_df.columns:
                        imputation_values = median_df["Median"].to_dict()
                    print(f"Loaded {len(imputation_values)} imputation medians from: '{imp_cand}'")
                    break
                except Exception as e:
                    print(f"Warning: Could not load imputation parquet '{imp_cand}': {e}")

    if retained_features is None:
        feat_candidates = [feature_order_path] if feature_order_path else []
        for s_base in search_base_dirs:
            for t_alias in target_aliases:
                feat_candidates.extend([
                    _join_uri(s_base, f"final_model_features_{t_alias}.json"),
                    _join_uri(s_base, "models", f"final_model_features_{t_alias}.json")
                ])
        for f_cand in feat_candidates:
            if f_cand and fs_model.exists(f_cand):
                retained_features = load_json(f_cand)
                print(f"Loaded feature order list from: '{f_cand}'")
                break

    if retained_features is None:
        raise FileNotFoundError(f"CRITICAL: Could not resolve feature order list for '{target_var}' in '{model_dir}' or parent directories.")

    print(f"Model configured: {len(retained_features)} features required.")

    # Apply Imputation and Vectorized Transformations
    work_df = input_df.copy()
    if imputation_values:
        for feat in retained_features:
            if feat in work_df.columns:
                work_df[feat] = work_df[feat].fillna(imputation_values.get(feat, 0.0))

    for col in retained_features:
        if col not in work_df.columns:
            print(f"  WARNING: Missing feature '{col}' in input data. Filling with 0.0.")
            work_df[col] = 0.0

    if transform_map or quantile_bounds:
        work_df, _ = _apply_feature_transformations(
            df=work_df,
            feature_columns=retained_features,
            transform_map=transform_map,
            quantile_bounds=quantile_bounds
        )

    # Model Execution
    if is_roughness:
        ensemble_candidates = []
        for s_base in search_base_dirs:
            ensemble_candidates.extend([
                _join_uri(s_base, "ensemble"),
                s_base,
                _join_uri(s_base, "n_out", "ensemble"),
                _join_uri(s_base, f"{target_var}_out", "ensemble")
            ])
        ensemble_candidates = list(dict.fromkeys(ensemble_candidates))

        ensemble_dir = None
        for e_cand in ensemble_candidates:
            if fs_model.exists(e_cand):
                files = [f for f in fs_model.ls(e_cand) if f.endswith((".pickle.dat", ".pkl", ".pickle")) and not os.path.basename(f).startswith(".")]
                if len(files) > 0:
                    ensemble_dir = e_cand
                    break

        if not ensemble_dir:
            raise FileNotFoundError(f"CRITICAL: No ensemble model pickles found for '{target_var}' in '{model_dir}' or searched paths: {ensemble_candidates}")

        ensemble_models = {}
        all_files = sorted(fs_model.ls(ensemble_dir))
        for fpath in all_files:
            fname = os.path.basename(fpath)
            if fname.endswith((".pickle.dat", ".pkl", ".pickle")) and not fname.startswith("."):
                full_p = fpath if ensemble_dir.startswith("s3://") else fpath
                m_name = os.path.splitext(fname)[0].replace(".pickle", "")
                ensemble_models[m_name] = load_pickle(full_p if ensemble_dir.startswith("s3://") else os.path.join(ensemble_dir, fname))

        num_models = len(ensemble_models)
        print(f"Loaded {num_models} ensemble models for roughness inference from '{ensemble_dir}'.")

        X_mat = work_df[retained_features].values
        Y_orig_matrix = np.zeros((len(work_df), num_models), dtype=np.float32)

        start_t = time.time()
        for idx, (m_name, model_obj) in enumerate(tqdm(ensemble_models.items(), desc=f"Predicting Ensemble {target_var}")):
            raw_preds = predict_model_fast(model_obj, X_mat, batch_size=32768)
            m_mse = model_val_mses.get(m_name, log_mse_val)
            m_bias = m_mse / 2.0 if transform_target == "log" else 0.0

            if transform_target == "log":
                Y_orig_matrix[:, idx] = np.clip(np.exp(raw_preds + m_bias), target_min_clip, target_max_clip)
            else:
                Y_orig_matrix[:, idx] = np.clip(raw_preds + m_bias, target_min_clip, target_max_clip)

        elapsed = time.time() - start_t
        print(f"Ensemble predictions completed in {elapsed:.2f}s.")

        print("Applying GMRF Topological Mainstem Regularization...")
        regularized_results_df = regularize_mainstem_roughness_ensemble(
            flowlines_df=work_df,
            ensemble_preds_matrix=Y_orig_matrix,
            lambda_single=lambda_single,
            gamma_reg=gamma_reg,
            fallback_n_single=0.035,
            comid_col_name=comid_col_name,
            mainstem_col_name=mainstem_col_name,
            toid_col_name=toid_col_name
        )

        pred_cols = [c for c in regularized_results_df.columns if c.startswith("pred_m_")]
        regularized_matrix = regularized_results_df[pred_cols].values

        regularized_results_df["n_reg_mean"] = np.mean(regularized_matrix, axis=1)
        regularized_results_df["n_reg_std"] = np.std(regularized_matrix, axis=1)
        regularized_results_df["n_reg_q10"] = np.percentile(regularized_matrix, 10, axis=1)
        regularized_results_df["n_reg_q25"] = np.percentile(regularized_matrix, 25, axis=1)
        regularized_results_df["n_reg_q50"] = np.percentile(regularized_matrix, 50, axis=1)
        regularized_results_df["n_reg_q75"] = np.percentile(regularized_matrix, 75, axis=1)
        regularized_results_df["n_reg_q90"] = np.percentile(regularized_matrix, 90, axis=1)

        conf_stats = calculate_ensemble_confidence(regularized_matrix, lower_ci_pct=lower_ci_pct, upper_ci_pct=upper_ci_pct)
        regularized_results_df[f"{target_var}_median"] = conf_stats["median"]
        regularized_results_df["confidence_score"] = conf_stats["confidence_score"]

        del X_mat, Y_orig_matrix, work_df, regularized_matrix, conf_stats
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return regularized_results_df

    else:
        m_candidates = []
        for s_base in search_base_dirs:
            if model_filename:
                m_candidates.append(_join_uri(s_base, model_filename))
            m_candidates.extend([
                _join_uri(s_base, f"trained_xgboost_model_update_{target_var}_final.pickle.dat"),
                _join_uri(s_base, f"trained_xgboost_model_update_{target_var}.pickle.dat"),
                _join_uri(s_base, f"{target_var}.pickle.dat"),
                _join_uri(s_base, f"{target_var}_final.pickle.dat")
            ])
        m_candidates = list(dict.fromkeys([c for c in m_candidates if c]))
        
        m_full_path = None
        for cand in m_candidates:
            if fs_model.exists(cand):
                m_full_path = cand
                break

        if not m_full_path:
            ls_files = [f for f in fs_model.ls(clean_dir) if f.endswith((".pickle.dat", ".pkl", ".pickle"))]
            if ls_files:
                m_full_path = ls_files[0] if clean_dir.startswith("s3://") else ls_files[0]

        if not m_full_path:
            raise FileNotFoundError(f"Model artifact not found for '{target_var}' in '{model_dir}'. Searched: {m_candidates}")

        print(f"Loading single model artifact from: '{m_full_path}'")
        trained_model = load_pickle(m_full_path)

        predictions_list = []
        comids_list = []

        chunk_indices = list(range(0, len(work_df), chunk_size))
        for i in tqdm(chunk_indices, desc=f"Predicting {target_var}"):
            chunk_df = work_df.iloc[i : i + chunk_size]
            if chunk_df.empty:
                continue

            comids = chunk_df[comid_col_name].values
            X_predict = chunk_df[retained_features]

            preds = predict_model_fast(trained_model, X_predict.values, batch_size=32768)

            if transform_target == "log1p":
                preds_orig = np.maximum(0.0, np.exp(preds + log_mse_val / 2.0) - 1.0)
            elif transform_target == "log":
                preds_orig = np.maximum(0.0, np.exp(preds + log_mse_val / 2.0))
            else:
                preds_orig = np.maximum(0.0, preds)

            if target_var == "r":
                preds_orig = np.clip(preds_orig, 1.0, None)

            predictions_list.append(preds_orig)
            comids_list.append(comids)

        out_df = pd.DataFrame({
            comid_col_name: np.concatenate(comids_list),
            "prediction": np.concatenate(predictions_list)
        })

        del work_df
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return out_df