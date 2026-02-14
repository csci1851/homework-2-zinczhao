"""Data loader for Homework 2: Ensemble Methods"""

import os
import pandas as pd
import numpy as np
from typing import Tuple
from pathlib import Path

class HW2DataLoader:
    def __init__(self):
        """Initialize data loader with cache directory for datasets"""
        # Store the homework root so we can read shared data files.
        self.homework_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = self.homework_dir.parent / "data"
        os.makedirs(self.data_dir, exist_ok=True)

    def get_cancer_genomics_data(
        self, csv_path: str = None, labels_path: str = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load the cancer genomics dataset from UCI after feature selection and data cleaning.

        Args:
            data_path: Path to the CSV file
            labels_path: Path to the labels file

        Returns:
            X: Features DataFrame
            y: Target Series (presence of cancer)
        """
        try:
            if csv_path is None:
                csv_path = self.data_dir / "cancer_genomics.csv"
            if labels_path is None:
                labels_path = self.data_dir / "labels_cancer_genomics.csv"

            labels = pd.read_csv(labels_path)
            data = pd.read_csv(csv_path)

            # Drop columns with missing values
            cleaned = data.dropna(axis=1)
            X = cleaned
            y = labels["Class"]

            return X, y

        except Exception as e:
            print(f"Error loading cancer genomics data: {e}")
            return None, None

    def get_heart_disease_data(self, csv_path=None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load the Heart Disease dataset.

        Args:
            csv_path: Path to the CSV file.

        Returns:
            X: Features DataFrame
            y: Target Series (presence of heart disease)
        """
        try:
            data = pd.read_csv(csv_path)
            print(f"Successfully loaded heart disease data with {len(data)} rows")

            target_col = "target"
            X = data.drop(target_col, axis=1)
            y = pd.Series(data[target_col], name=target_col)

            return X, y
        except Exception as e:
            print(f"Error loading heart disease data: {e}")
            return None, None
