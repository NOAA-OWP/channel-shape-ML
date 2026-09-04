import logging
import os
import sys
import fsspec


class PipelineLogger:
    """
    Dual-channel logging engine streaming formatted logs to stdout and
    persisting them to a log file on local disk or AWS S3 with resilient fallback.
    """
    def __init__(self, log_file_path: str = None, level: int = logging.INFO):
        self.logger = logging.getLogger("RIver_ML_Pipeline")
        self.logger.setLevel(level)
        self.logger.handlers.clear()
        
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Console Stream Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Resilient File Stream Handler
        self.remote_s3_path = None
        self.local_log_file = None
        
        if log_file_path:
            if log_file_path.startswith("s3://"):
                self.remote_s3_path = log_file_path
                self.local_log_file = os.path.join("/tmp/river_ml", "river_ml_" + os.path.basename(log_file_path))
            else:
                self.local_log_file = log_file_path

            try:
                log_dir = os.path.dirname(os.path.abspath(self.local_log_file))
                os.makedirs(log_dir, exist_ok=True)
                file_handler = logging.FileHandler(self.local_log_file, mode="w", encoding="utf-8")
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
            except (PermissionError, OSError) as e:
                # Fallback to writeable /tmp directory so pipeline never crashes on secondary logging failure
                fallback_path = os.path.join("/tmp/river_ml", "river_ml_execution.log")
                os.makedirs("/tmp/river_ml", exist_ok=True)
                sys.stderr.write(f"[LOGGING WARNING] Could not write to primary log path '{self.local_log_file}' ({e}). Falling back to '{fallback_path}'.\n")
                try:
                    self.local_log_file = fallback_path
                    file_handler = logging.FileHandler(self.local_log_file, mode="w", encoding="utf-8")
                    file_handler.setFormatter(formatter)
                    self.logger.addHandler(file_handler)
                except Exception as e_fallback:
                    sys.stderr.write(f"[LOGGING ERROR] Fallback file logging also failed ({e_fallback}). Console logging will remain active.\n")

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    def flush_to_remote(self):
        """Flushes local log buffer and syncs to S3 bucket if remote URI is specified."""
        if self.remote_s3_path and self.local_log_file and os.path.exists(self.local_log_file):
            try:
                fs, _ = fsspec.core.url_to_fs(self.remote_s3_path)
                with open(self.local_log_file, "rb") as f_src:
                    with fs.open(self.remote_s3_path, "wb") as f_dst:
                        f_dst.write(f_src.read())
            except Exception as e:
                self.logger.warning(f"Could not sync execution log to S3 ({self.remote_s3_path}): {e}")