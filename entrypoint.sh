#!/usr/bin/env bash
set -e

umask 000

# Export performance flags for GDAL, S3, and OpenBLAS
export GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR
export CPL_VSIL_CURL_ALLOWED_EXTENSIONS=.gpkg,.parquet,.json
export AWS_S3_MAX_CONCURRENT_REQUESTS=${AWS_S3_MAX_CONCURRENT_REQUESTS:-64}
export PYARROW_IGNORE_TIMEZONE=1

# Execute inference runner
exec python /app/run_inference.py "$@"