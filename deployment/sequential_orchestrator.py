import gc
import os
import fsspec
import geopandas as gpd
import numpy as np
import pandas as pd
import torch

from deployment.inference_engine import run_stage_inference
from src.core.io import load_geopackage, load_parquet, save_parquet
from src.core.logging import PipelineLogger
from src.features.pipeline import engineer_flowline_features


def run_full_sequential_inference(
    flowlines_gpkg_path: str,
    slopes_parquet_path: str,
    output_base_dir: str,
    stage_model_paths: dict,
    process_domain: str = "custom_domain",
    comid_col_name: str = "flowpath_id",
    mainstem_col_name: str = "mainstemlp",
    toid_col_name: str = "flowpath_toid",
    flowpath_layer: str = "flowpaths",
    chunk_size: int = 150000,
    lambda_single: float = 10.0,
    gamma_reg: float = 0.01,
    debug_mode: bool = False,
    logger: PipelineLogger = None
) -> pd.DataFrame:
    """
    Executes end-to-end sequential hydrological inference using explicit model paths for all 6 stages.
    """
    if logger is None:
        log_file = f"{output_base_dir}/{process_domain}/deployment/data/river_ml_{process_domain}.log"
        logger = PipelineLogger(log_file_path=log_file)

    logger.info("=" * 100)
    logger.info(f"STARTING RIVER ML SEQUENTIAL INFERENCE PIPELINE: {process_domain.upper()}")
    logger.info(f"Execution Config: Debug Mode = {debug_mode}, Chunk Size = {chunk_size:,}")
    logger.info("=" * 100)

    logger.info(f"Ingesting Network Flowlines from: '{flowlines_gpkg_path}'...")
    flowlines = load_geopackage(flowlines_gpkg_path, layer=flowpath_layer)
    
    flowlines[comid_col_name] = flowlines[comid_col_name].astype(int)
    flowlines[toid_col_name] = flowlines[toid_col_name].astype(int)
    flowlines["streamorder"] = flowlines["streamorder"].astype("Int64")
    flowlines["terminalfl"] = flowlines["terminalfl"].astype("Int64")
    flowlines[mainstem_col_name] = flowlines[mainstem_col_name].astype("Int64")

    logger.info(f"Loading Slope Data from: '{slopes_parquet_path}'...")
    ax_data = load_parquet(slopes_parquet_path)
    if 'final_smoothed_slope' in ax_data.columns and 'slope' not in ax_data.columns:
        ax_data.rename(columns={'final_smoothed_slope': 'slope'}, inplace=True)
    elif 'slope_m_m' in ax_data.columns and 'slope' not in ax_data.columns:
        ax_data.rename(columns={'slope_m_m': 'slope'}, inplace=True)
    if 'id' in ax_data.columns and comid_col_name not in ax_data.columns:
        ax_data.rename(columns={'id': comid_col_name}, inplace=True)

    base_cols = [comid_col_name, toid_col_name, "hydroseq", mainstem_col_name, "pathlength",
                 "terminalfl", "streamorder", "arb_sum", "lengthkm", "areasqkm", "totdasqkm", "geometry"]
    infer_df = flowlines[[c for c in base_cols if c in flowlines.columns]].copy()
    infer_df = infer_df.merge(ax_data[[comid_col_name, "slope"]], on=comid_col_name, how="left")
    infer_df = infer_df.drop_duplicates(subset=[comid_col_name], keep="first").reset_index(drop=True)

    slopes_lookup = infer_df[[comid_col_name, "slope"]].copy()
    slopes_lookup["slope"] = slopes_lookup["slope"].astype(np.float32)

    del flowlines, ax_data
    gc.collect()

    tw_path = stage_model_paths.get("TW_bf_m")
    logger.info("-" * 90)
    logger.info(f"STAGE 1: Predicting Channel Top Width (TW_bf_m)...")
    logger.info(f"Model Path: {tw_path}")
    logger.info("-" * 90)
    infer_tw = engineer_flowline_features(infer_df, target_stage="tw_bf")
    
    tw_preds_df = run_stage_inference(
        target_var="TW_bf_m",
        input_df=infer_tw,
        model_dir=tw_path,
        comid_col_name=comid_col_name,
        chunk_size=chunk_size,
        transform_target="log1p",
        is_roughness=False
    )
    tw_preds_df.rename(columns={"prediction": "owp_tw_bf_m"}, inplace=True)
    tw_preds_df["owp_tw_bf_m"] = tw_preds_df["owp_tw_bf_m"].astype(np.float32)
    tw_out_path = f"{output_base_dir}/{process_domain}/deployment/data/TW_bf_m_out/processed/tw_bf_m.parquet"
    save_parquet(tw_preds_df, tw_out_path)
    logger.info(f"--> Saved Stage 1 TW predictions to: '{tw_out_path}'")

    infer_df = infer_df.merge(tw_preds_df[[comid_col_name, "owp_tw_bf_m"]], on=comid_col_name, how="left")
    infer_df.rename(columns={"owp_tw_bf_m": "tw_bf_pred"}, inplace=True)

    del infer_tw
    gc.collect()

    y_path = stage_model_paths.get("Y_bf_m")
    logger.info("-" * 90)
    logger.info(f"STAGE 2: Predicting Bankfull Channel Depth (Y_bf_m)...")
    logger.info(f"Model Path: {y_path}")
    logger.info("-" * 90)
    infer_y = engineer_flowline_features(infer_df, target_stage="y_bf")
    
    y_preds_df = run_stage_inference(
        target_var="Y_bf_m",
        input_df=infer_y,
        model_dir=y_path,
        comid_col_name=comid_col_name,
        chunk_size=chunk_size,
        transform_target="log1p",
        is_roughness=False
    )
    y_preds_df.rename(columns={"prediction": "owp_y_bf_m"}, inplace=True)
    y_preds_df["owp_y_bf_m"] = y_preds_df["owp_y_bf_m"].astype(np.float32)
    y_out_path = f"{output_base_dir}/{process_domain}/deployment/data/Y_bf_m_out/processed/y_bf_m.parquet"
    save_parquet(y_preds_df, y_out_path)
    logger.info(f"--> Saved Stage 2 Depth predictions to: '{y_out_path}'")

    infer_df = infer_df.merge(y_preds_df[[comid_col_name, "owp_y_bf_m"]], on=comid_col_name, how="left")
    infer_df.rename(columns={"owp_y_bf_m": "y_bf_pred"}, inplace=True)

    del infer_y
    gc.collect()

    r_path = stage_model_paths.get("r")
    logger.info("-" * 90)
    logger.info(f"STAGE 3: Predicting Dingman Channel Shape Exponent (r)...")
    logger.info(f"Model Path: {r_path}")
    logger.info("-" * 90)
    infer_r = engineer_flowline_features(infer_df, target_stage="r")
    
    r_preds_df = run_stage_inference(
        target_var="r",
        input_df=infer_r,
        model_dir=r_path,
        comid_col_name=comid_col_name,
        chunk_size=chunk_size,
        transform_target="log1p",
        is_roughness=False
    )
    r_preds_df.rename(columns={"prediction": "owp_r_bf"}, inplace=True)
    r_preds_df["owp_r_bf"] = r_preds_df["owp_r_bf"].clip(lower=1.0).astype(np.float32)
    r_out_path = f"{output_base_dir}/{process_domain}/deployment/data/r_out/processed/r_bf.parquet"
    save_parquet(r_preds_df, r_out_path)
    logger.info(f"--> Saved Stage 3 Dingman r predictions to: '{r_out_path}'")

    infer_df = infer_df.merge(r_preds_df[[comid_col_name, "owp_r_bf"]], on=comid_col_name, how="left")
    infer_df.rename(columns={"owp_r_bf": "r_bf_pred"}, inplace=True)

    del infer_r
    gc.collect()

    logger.info("-" * 90)
    logger.info("STAGE 4, 5, 6: Feature Engineering for Roughness Stages...")
    logger.info("-" * 90)
    infer_n = engineer_flowline_features(infer_df, target_stage="n")

    del infer_df
    gc.collect()

    roughness_stages = [
        ("n", "owp_n_single", "n_single_out"),
        ("n_in_channel", "owp_n_in_channel", "n_in_channel_out"),
        ("n_out_channel", "owp_n_out_channel", "n_out_channel_out")
    ]

    roughness_results = {}

    for target_var, std_col_name, out_folder in roughness_stages:
        n_model_path = stage_model_paths.get(target_var)
        logger.info(f"\nRunning Ensemble Inference + GMRF Regularization for Roughness: '{target_var}'...")
        logger.info(f"Model Path: {n_model_path}")
        try:
            regularized_df = run_stage_inference(
                target_var=target_var,
                input_df=infer_n,
                model_dir=n_model_path,
                comid_col_name=comid_col_name,
                mainstem_col_name=mainstem_col_name,
                toid_col_name=toid_col_name,
                chunk_size=chunk_size,
                transform_target="log",
                is_roughness=True,
                lambda_single=lambda_single,
                gamma_reg=gamma_reg
            )

            raw_save_path = f"{output_base_dir}/{process_domain}/deployment/data/{out_folder}/regularized_predictions_{target_var}.parquet"
            save_parquet(regularized_df, raw_save_path)

            conf_col_name = f"confidence_score_{target_var}"
            final_clean = pd.DataFrame({
                comid_col_name: regularized_df[comid_col_name].astype("Int64"),
                std_col_name: regularized_df[f"{target_var}_median"].astype(np.float32),
                f"{std_col_name}_lcl": regularized_df["n_reg_q25"].astype(np.float32),
                f"{std_col_name}_ucl": regularized_df["n_reg_q75"].astype(np.float32),
                conf_col_name: regularized_df["confidence_score"].astype(np.float32)
            })

            final_save_path = f"{output_base_dir}/{process_domain}/deployment/data/{out_folder}/processed/{target_var}_raw.parquet"
            save_parquet(final_clean, final_save_path)
            logger.info(f"--> Successfully saved Final Roughness ({target_var}) to: '{final_save_path}'")

            roughness_results[target_var] = final_clean

            del regularized_df
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"[STAGE ERROR] Roughness stage '{target_var}' encountered an error: {e}")

    del infer_n
    gc.collect()

    logger.info("=" * 100)
    logger.info("CONSOLIDATING ALL 6 HYDRAULIC & ROUGHNESS PREDICTIONS INTO MASTER DATASET")
    logger.info("=" * 100)

    merged_df = slopes_lookup.merge(tw_preds_df[[comid_col_name, "owp_tw_bf_m"]], on=comid_col_name, how="left") \
        .merge(y_preds_df[[comid_col_name, "owp_y_bf_m"]], on=comid_col_name, how="left") \
        .merge(r_preds_df[[comid_col_name, "owp_r_bf"]], on=comid_col_name, how="left")

    if "n" in roughness_results:
        merged_df = merged_df.merge(roughness_results["n"], on=comid_col_name, how="left")
    if "n_in_channel" in roughness_results:
        merged_df = merged_df.merge(roughness_results["n_in_channel"], on=comid_col_name, how="left")
    if "n_out_channel" in roughness_results:
        merged_df = merged_df.merge(roughness_results["n_out_channel"], on=comid_col_name, how="left")

    merged_df[comid_col_name] = merged_df[comid_col_name].astype("Int64")

    master_output_path = f"{output_base_dir}/{process_domain}/deployment/data/final_consolidated_predictions_{process_domain}.parquet"
    save_parquet(merged_df, master_output_path)
    logger.info(f"--> Master Consolidated Dataset saved ({len(merged_df):,} reaches, {len(merged_df.columns)} columns) to:")
    logger.info(f"    '{master_output_path}'")

    if not debug_mode:
        logger.info("[CLEANUP] Debug mode is OFF. Removing intermediate stage directories to save disk space...")
        fs, _ = fsspec.core.url_to_fs(output_base_dir)
        stage_dirs_to_clean = [
            f"{output_base_dir}/{process_domain}/deployment/data/TW_bf_m_out",
            f"{output_base_dir}/{process_domain}/deployment/data/Y_bf_m_out",
            f"{output_base_dir}/{process_domain}/deployment/data/r_out",
            f"{output_base_dir}/{process_domain}/deployment/data/n_single_out",
            f"{output_base_dir}/{process_domain}/deployment/data/n_in_channel_out",
            f"{output_base_dir}/{process_domain}/deployment/data/n_out_channel_out"
        ]
        for s_dir in stage_dirs_to_clean:
            try:
                if fs.exists(s_dir):
                    fs.rm(s_dir, recursive=True)
                    logger.info(f"  • Removed intermediate directory: {s_dir}")
            except Exception as e_clean:
                logger.warning(f"  • Could not remove {s_dir}: {e_clean}")
        logger.info(f"--> Clean complete. Only master dataset retained: '{master_output_path}'")

    del slopes_lookup, tw_preds_df, y_preds_df, r_preds_df, roughness_results
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("=" * 100)
    logger.info(f"RIVER ML SEQUENTIAL INFERENCE PIPELINE COMPLETED SUCCESSFULLY FOR: {process_domain.upper()}")
    logger.info("=" * 100)

    logger.flush_to_remote()
    return merged_df