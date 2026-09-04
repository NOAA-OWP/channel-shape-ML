import io
import json
import os
import sys
import pickle
import fsspec
import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
import src.models.architectures as architectures
import src.models.wrappers as wrappers

class RobustUnpickler(pickle.Unpickler):
    """
    Robust Unpickler that safely resolves models pickled under '__main__'
    or moved module namespaces to their correct class definitions in src.models.
    """
    def find_class(self, module, name):
        if name in [
            'PyTorchTabularWrapper',
            'PhysicalResTabNet',
            'ModernResTabBlockV4',
            'PiecewiseLinearEncoding',
            'CrossFeatureInteraction',
            'RMSNorm',
            'SqueezeAndExcitation1D'
        ]:
            try:
                if hasattr(wrappers, name):
                    return getattr(wrappers, name)
            except Exception:
                pass
            try:
                if hasattr(architectures, name):
                    return getattr(architectures, name)
            except Exception:
                pass

        try:
            return super().find_class(module, name)
        except (AttributeError, ModuleNotFoundError, ImportError):
            if module == '__main__':
                main_mod = sys.modules.get('__main__')
                if main_mod and hasattr(main_mod, name):
                    return getattr(main_mod, name)
            raise


def open_file_stream(file_path: str, mode: str = "rb"):
    """Opens a file stream from either a local filesystem or AWS S3 URI (s3://...)."""
    return fsspec.open(file_path, mode=mode)


def load_parquet(file_path: str, columns: list = None) -> pd.DataFrame:
    """Reads a Parquet file from local disk or S3 with optional column projection."""
    return pd.read_parquet(file_path, columns=columns)


def save_parquet(df: pd.DataFrame, file_path: str, index: bool = False) -> None:
    """Writes a DataFrame/GeoDataFrame to Parquet safely on local disk or S3."""
    if not file_path.startswith("s3://"):
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
    if isinstance(df, gpd.GeoDataFrame):
        df.to_parquet(file_path, index=index)
    elif 'geometry' in df.columns:
        # If standard DataFrame contains GeometryDtype, promote to GeoDataFrame or drop
        try:
            gdf = gpd.GeoDataFrame(df, geometry='geometry')
            gdf.to_parquet(file_path, index=index)
        except Exception:
            df.drop(columns=['geometry']).to_parquet(file_path, index=index, engine="pyarrow")
    else:
        df.to_parquet(file_path, index=index, engine="pyarrow")


def load_geopackage(file_path: str, layer: str = None) -> gpd.GeoDataFrame:
    """Reads a GeoPackage layer from local disk or S3 via fsspec."""
    if file_path.startswith("s3://"):
        with fsspec.open(file_path, mode="rb") as f:
            return gpd.read_file(f, layer=layer)
    return gpd.read_file(file_path, layer=layer)


def load_json(file_path: str) -> dict:
    """Loads JSON metadata from local disk or S3."""
    with fsspec.open(file_path, mode="r") as f:
        return json.load(f)


def save_json(data: dict, file_path: str, indent: int = 4) -> None:
    """Saves JSON dictionary to local disk or S3."""
    if not file_path.startswith("s3://"):
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    with fsspec.open(file_path, mode="w") as f:
        json.dump(data, f, indent=indent)


def load_pickle(file_path: str):
    """Loads a pickled model object from local disk or S3 using RobustUnpickler."""
    with fsspec.open(file_path, mode="rb") as f:
        data = f.read()
        return RobustUnpickler(io.BytesIO(data)).load()


def save_pickle(obj, file_path: str) -> None:
    """Saves a model object to a pickle file on local disk or S3."""
    if not file_path.startswith("s3://"):
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    with fsspec.open(file_path, mode="wb") as f:
        pickle.dump(obj, f)