import argparse
import json
import os
import sys

from deployment.sequential_orchestrator import run_full_sequential_inference
from src.core.logging import PipelineLogger
from src.models.architectures import (
    CrossFeatureInteraction,
    ModernResTabBlockV4,
    PhysicalResTabNet,
    PiecewiseLinearEncoding,
    RMSNorm,
    SqueezeAndExcitation1D
)
from src.models.wrappers import PyTorchTabularWrapper


def validate_output_directory(output_dir: str):
    """Pre-flight check to ensure the output destination is writeable."""
    if output_dir.startswith("s3://"):
        return
    try:
        os.makedirs(output_dir, exist_ok=True)
        probe_file = os.path.join(output_dir, ".write_test_probe")
        with open(probe_file, "w") as f:
            f.write("probe")
        os.remove(probe_file)
    except Exception as e:
        sys.stderr.write(
            f"\n[CRITICAL ERROR] Output directory '{output_dir}' is not writeable.\n"
            f"Details: {e}\n"
            f"Please verify host folder permissions.\n\n"
        )
        raise


def parse_args():
    parser = argparse.ArgumentParser(
        description="RIver ML Pipeline: Sequential Inference Engine (Local and S3 Compatible)"
    )
    
    parser.add_argument("--flowlines_path", type=str, required=True, help="Explicit local path or s3:// URI to Flowpaths GeoPackage (.gpkg)")
    parser.add_argument("--slopes_path", type=str, required=True, help="Explicit local path or s3:// URI to slope parquet file")
    parser.add_argument("--output_dir", type=str, required=True, help="Explicit local path or s3:// URI base directory to store predictions")
    parser.add_argument("--process_domain", type=str, required=True, help="Domain identifier used for output naming (e.g. superconus, ak, hi, prvi, guam)")

    parser.add_argument("--tw_model_path", type=str, required=True, help="Full explicit path to Stage 1 Top Width model directory")
    parser.add_argument("--y_model_path", type=str, required=True, help="Full explicit path to Stage 2 Depth model directory")
    parser.add_argument("--r_model_path", type=str, required=True, help="Full explicit path to Stage 3 Dingman r model directory")
    parser.add_argument("--n_model_path", type=str, required=True, help="Full explicit path to Stage 4 Single Roughness model directory")
    parser.add_argument("--n_in_channel_model_path", type=str, required=True, help="Full explicit path to Stage 5 In-Channel Roughness model directory")
    parser.add_argument("--n_out_channel_model_path", type=str, required=True, help="Full explicit path to Stage 6 Overbank Roughness model directory")
    
    parser.add_argument("--aws_profile", type=str, default=None, help="AWS Profile name configured via 'aws sso login'")
    
    parser.add_argument("--flowpath_layer", type=str, default="flowpaths", help="Layer name inside the Flowlines GeoPackage")
    parser.add_argument("--comid_col", type=str, default="flowpath_id", help="Reach ID column name")
    parser.add_argument("--mainstem_col", type=str, default="mainstemlp", help="Mainstem level path topological column name")
    parser.add_argument("--toid_col", type=str, default="flowpath_toid", help="Downstream target ID column name")
    
    parser.add_argument("--chunk_size", type=int, default=100000, help="Inference batch chunk size for RAM/VRAM safety")
    parser.add_argument("--lambda_reg", "--lambda_smooth", dest="lambda_reg", type=float, default=10.0, help="GMRF topological regularization smoothness weight (lambda)")
    parser.add_argument("--gamma_reg", type=float, default=0.01, help="GMRF prior attraction regularization weight (gamma)")
    parser.add_argument("--debug_mode", "--debug", dest="debug_mode", action="store_true", default=False, help="Retain intermediate stage files (default: False, purges intermediate files)")
    parser.add_argument("--log_file", type=str, default=None, help="Explicit local or s3:// path for execution log.out file")
    parser.add_argument("--stage_model_paths", type=str, default=None, help="JSON string mapping stage targets to full model directory paths")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    validate_output_directory(args.output_dir)

    if args.aws_profile:
        os.environ["AWS_PROFILE"] = args.aws_profile

    log_path = args.log_file or f"{args.output_dir}/{args.process_domain}/deployment/data/river_ml_{args.process_domain}.log"
    logger = PipelineLogger(log_file_path=log_path)

    stage_model_paths = {
        "TW_bf_m": args.tw_model_path,
        "Y_bf_m": args.y_model_path,
        "r": args.r_model_path,
        "n": args.n_model_path,
        "n_in_channel": args.n_in_channel_model_path,
        "n_out_channel": args.n_out_channel_model_path
    }

    if args.stage_model_paths:
        try:
            custom_paths = json.loads(args.stage_model_paths)
            stage_model_paths.update(custom_paths)
        except Exception as e:
            logger.warning(f"Failed to parse --stage_model_paths JSON ({e}). Using CLI flags.")

    logger.info("RIver ML Pipeline initialized.")
    if "AWS_PROFILE" in os.environ:
        logger.info(f"Active AWS Profile: '{os.environ['AWS_PROFILE']}'")
    logger.info(f"Target Process Domain: '{args.process_domain}'")
    logger.info("Explicit Model Paths Configured:")
    for stage_key, path_val in stage_model_paths.items():
        logger.info(f"  • {stage_key:15s} --> {path_val}")

    run_full_sequential_inference(
        flowlines_gpkg_path=args.flowlines_path,
        slopes_parquet_path=args.slopes_path,
        output_base_dir=args.output_dir,
        stage_model_paths=stage_model_paths,
        process_domain=args.process_domain,
        comid_col_name=args.comid_col,
        mainstem_col_name=args.mainstem_col,
        toid_col_name=args.toid_col,
        flowpath_layer=args.flowpath_layer,
        chunk_size=args.chunk_size,
        lambda_single=args.lambda_reg,
        gamma_reg=args.gamma_reg,
        debug_mode=args.debug_mode,
        logger=logger
    )