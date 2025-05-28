from pathlib import Path

import arviz as az
import joblib


class ModelLoader:
    """A class to load and manage a Bayesian model and its associated scaler."""

    def __init__(
        self,
        base_dir=None,
        model_filename="multi_outcome_model.nc",
        scaler_filename="pa_outcome_scaler.joblib",
    ):
        """
        Initialize the ModelLoader with paths to the model and scaler files.

        Parameters
        ----------
        base_dir : str or Path, optional
            Base directory where model files are stored. If None, current directory is used.
        model_filename : str
            Filename of the model file in NetCDF format
        scaler_filename : str
            Filename of the joblib scaler file

        """
        # Handle base directory
        if base_dir is None:
            self.base_dir = Path.cwd()
        else:
            self.base_dir = Path(base_dir)

        # Store filenames
        self.model_filename = model_filename
        self.scaler_filename = scaler_filename

        # Build full paths
        self.model_path = self.base_dir / model_filename
        self.scaler_path = self.base_dir / scaler_filename

        # Initialize model and scaler
        self.model = None
        self._scaler = None

    def set_paths(self, base_dir=None, model_filename=None, scaler_filename=None):
        """
        Update the file paths.

        Parameters
        ----------
        base_dir : str or Path, optional
            New base directory
        model_filename : str, optional
            New model filename
        scaler_filename : str, optional
            New scaler filename

        """
        if base_dir is not None:
            self.base_dir = Path(base_dir)

        if model_filename is not None:
            self.model_filename = model_filename

        if scaler_filename is not None:
            self.scaler_filename = scaler_filename

        # Rebuild full paths
        self.model_path = self.base_dir / self.model_filename
        self.scaler_path = self.base_dir / self.scaler_filename

        return self

    def load_model(self):
        """
        Load the Bayesian model using arviz.

        Returns
        -------
        arviz.InferenceData
            The loaded model

        """
        self.model = az.from_netcdf(self.model_path)

        return self.model

    def load_scaler(self):
        """
        Load the scaler using joblib.

        Returns
        -------
        object
            The loaded scaler

        """
        self._scaler = joblib.load(self.scaler_path)

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
        return self.model, self.scaler

    @property
    def idata(self):
        """Property to access the model data."""
        if self.model is None:
            self.load_model()
        return self.model

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
