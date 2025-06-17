from pathlib import Path
import os # Added
import io # Added
import boto3 # Added
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError # Added
import arviz as az
import joblib
import logging # Added for logging

logger = logging.getLogger(__name__) # Added for logging

# --- S3 Helper Functions (redefined for self-containment) ---
def is_s3_path(path: str) -> bool:
    """Checks if the given path is an S3 path."""
    return isinstance(path, str) and path.startswith("s3://")

def parse_s3_path(s3_path: str) -> tuple[str, str]:
    """Parses an S3 path into bucket name and object key."""
    if not is_s3_path(s3_path):
        raise ValueError(f"Invalid S3 path: {s3_path}. Must start with 's3://'")
    path_parts = s3_path.replace("s3://", "").split("/", 1)
    if len(path_parts) < 2 or not path_parts[1]: # Ensure there's a key
        raise ValueError(f"Invalid S3 path: {s3_path}. Must include bucket and a non-empty key.")
    bucket_name = path_parts[0]
    object_key = path_parts[1]
    return bucket_name, object_key

class ModelLoader:
    """A class to load and manage a Bayesian model and its associated scaler from local or S3 paths."""

    def __init__(
        self,
        base_dir=None, # Can be local path string, Path object, or S3 URI string
        model_filename="multi_outcome_model.nc",
        scaler_filename="pa_outcome_scaler.joblib",
    ):
        """
        Initialize the ModelLoader with paths to the model and scaler files.

        Parameters
        ----------
        base_dir : str or Path, optional
            Base directory where model files are stored (local path or S3 URI like 's3://my-bucket/models/').
            If None, current local directory is used.
        model_filename : str
            Filename of the model file (e.g., 'multi_outcome_model.nc')
        scaler_filename : str
            Filename of the joblib scaler file (e.g., 'pa_outcome_scaler.joblib')
        """
        self._model = None # Renamed to avoid conflict with property
        self._scaler = None
        self.set_paths(base_dir, model_filename, scaler_filename)


    def set_paths(self, base_dir=None, model_filename=None, scaler_filename=None):
        """
        Update the file paths. Paths can be local or S3 URIs.

        Parameters
        ----------
        base_dir : str or Path, optional
            New base directory or S3 URI.
        model_filename : str, optional
            New model filename.
        scaler_filename : str, optional
            New scaler filename.
        """
        if base_dir is not None:
            self.base_dir_str = str(base_dir) # Store as string
        elif not hasattr(self, 'base_dir_str') or self.base_dir_str is None: # Handle initial call from __init__
            self.base_dir_str = str(Path.cwd())


        if model_filename is not None:
            self.model_filename = model_filename
        # else: self.model_filename remains from __init__ or previous set_paths

        if scaler_filename is not None:
            self.scaler_filename = scaler_filename
        # else: self.scaler_filename remains from __init__ or previous set_paths

        # Rebuild full paths
        if is_s3_path(self.base_dir_str):
            s3_base = self.base_dir_str if self.base_dir_str.endswith('/') else self.base_dir_str + '/'
            # Using os.path.join for S3 key construction is not standard for URLs, simple concatenation is better.
            # Ensure filenames themselves don't have leading slashes if s3_base ensures trailing one.
            self.model_path = s3_base + self.model_filename.lstrip('/')
            self.scaler_path = s3_base + self.scaler_filename.lstrip('/')
        else:
            _base_path_obj = Path(self.base_dir_str)
            self.model_path = str(_base_path_obj / self.model_filename)
            self.scaler_path = str(_base_path_obj / self.scaler_filename)

        logger.info(f"Model path set to: {self.model_path}")
        logger.info(f"Scaler path set to: {self.scaler_path}")

        # Reset loaded components so they are reloaded from new paths
        self._model = None
        self._scaler = None
        return self

    def load_model(self):
        """
        Load the Bayesian model using arviz from local path or S3.

        Returns
        -------
        arviz.InferenceData
            The loaded model
        """
        logger.info(f"Loading model from: {self.model_path}")
        if is_s3_path(self.model_path):
            try:
                bucket_name, object_key = parse_s3_path(self.model_path)
                s3_client = boto3.client('s3')
                response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
                buffer = io.BytesIO(response['Body'].read())
                self._model = az.from_netcdf(buffer) # Assign to _model
                logger.info(f"Successfully loaded model from S3: {self.model_path}")
            except (NoCredentialsError, PartialCredentialsError):
                logger.error(f"AWS credentials not found or incomplete for S3 path: {self.model_path}", exc_info=True)
                raise
            except ClientError as e:
                if e.response['Error']['Code'] == 'NoSuchKey':
                    logger.error(f"S3 object not found for model at {self.model_path}: {e}", exc_info=True)
                else:
                    logger.error(f"AWS S3 ClientError loading model from {self.model_path}: {e}", exc_info=True)
                raise FileNotFoundError(f"Failed to load model from S3: {self.model_path}") from e
            except Exception as e:
                logger.error(f"Error loading model from S3 path {self.model_path}: {e}", exc_info=True)
                raise
        else: # Local file path
            local_p = Path(self.model_path)
            if not local_p.exists():
                logger.error(f"Model file not found at local path: {self.model_path}")
                raise FileNotFoundError(f"Model file not found at local path: {self.model_path}")
            try:
                self._model = az.from_netcdf(self.model_path) # Assign to _model
                logger.info(f"Successfully loaded model from local path: {self.model_path}")
            except Exception as e:
                logger.error(f"Error loading model from local path {self.model_path}: {e}", exc_info=True)
                raise
        return self._model

    def load_scaler(self):
        """
        Load the scaler using joblib from local path or S3.

        Returns
        -------
        object
            The loaded scaler
        """
        logger.info(f"Loading scaler from: {self.scaler_path}")
        if is_s3_path(self.scaler_path):
            try:
                bucket_name, object_key = parse_s3_path(self.scaler_path)
                s3_client = boto3.client('s3')
                response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
                buffer = io.BytesIO(response['Body'].read())
                self._scaler = joblib.load(buffer)
                logger.info(f"Successfully loaded scaler from S3: {self.scaler_path}")
            except (NoCredentialsError, PartialCredentialsError):
                logger.error(f"AWS credentials not found or incomplete for S3 path: {self.scaler_path}", exc_info=True)
                raise
            except ClientError as e:
                if e.response['Error']['Code'] == 'NoSuchKey':
                    logger.error(f"S3 object not found for scaler at {self.scaler_path}: {e}", exc_info=True)
                else:
                    logger.error(f"AWS S3 ClientError loading scaler from {self.scaler_path}: {e}", exc_info=True)
                raise FileNotFoundError(f"Failed to load scaler from S3: {self.scaler_path}") from e
            except Exception as e:
                logger.error(f"Error loading scaler from S3 path {self.scaler_path}: {e}", exc_info=True)
                raise
        else: # Local file path
            local_p = Path(self.scaler_path)
            if not local_p.exists():
                logger.error(f"Scaler file not found at local path: {self.scaler_path}")
                raise FileNotFoundError(f"Scaler file not found at local path: {self.scaler_path}")
            try:
                self._scaler = joblib.load(self.scaler_path)
                logger.info(f"Successfully loaded scaler from local path: {self.scaler_path}")
            except Exception as e:
                logger.error(f"Error loading scaler from local path {self.scaler_path}: {e}", exc_info=True)
                raise
        return self._scaler

    def load_all(self):
        """
        Load both the model and scaler.

        Returns
        -------
        tuple
            (model, scaler) tuple containing both loaded objects
        """
        self.load_model()
        self.load_scaler()
        return self._model, self._scaler # Return internal attributes

    @property
    def idata(self):
        """Property to access the model data."""
        if self._model is None:
            self.load_model()
        return self._model

    @property
    def scaler(self):
        """Property to access the scaler."""
        if self._scaler is None:
            self.load_scaler()
        return self._scaler

    @scaler.setter
    def scaler(self, value):
        """Setter for the scaler property."""
        self._scaler = value

# Example usage:
# loader = ModelLoader(base_dir='s3://my-ml-models/projectx/', model_filename='model_v1.nc', scaler_filename='scaler_v1.joblib')
# model_data = loader.idata
# scaler_obj = loader.scaler

# loader_local = ModelLoader(base_dir='/path/to/models', model_filename='model_v1.nc', scaler_filename='scaler_v1.joblib')
# model_data_local = loader_local.idata
# scaler_obj_local = loader_local.scaler
