# data_loader.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold  
from collections import Counter
from typing import List, Tuple
import logging



class IoMTDataLoader:
    def __init__(self, filepath: str, min_samples_per_class: int = 2):
        self.filepath = filepath
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.min_samples_per_class = min_samples_per_class
        self.attack_labels = []

    def load_data(self) -> pd.DataFrame:
        """Load and preprocess the IoMT dataset"""
        try:
            df = pd.read_csv(self.filepath)
            logging.info(f"Dataset loaded with shape: {df.shape}")
            return df
        except Exception as e:
            logging.error(f"Data loading error: {str(e)}")
            raise

    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Process features and labels using your dataset headers"""
        try:
            # Assuming last column is the attack label
            self.attack_labels = df.iloc[:, -1].values.tolist()
            X = df.iloc[:, :-1].values  # Features
            y = df.iloc[:, -1].values   # Labels

            # Handle class imbalance
            X, y = self._handle_imbalanced_data(X, y)

            # Normalize features
            X = self.scaler.fit_transform(X)

            # Encode labels
            y = self.label_encoder.fit_transform(y)

            return X, y, self.attack_labels
        except Exception as e:
            logging.error(f"Preprocessing error: {str(e)}")
            raise

    def _handle_imbalanced_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Balance classes using the specified minimum samples"""
        class_counts = Counter(y)
        valid_classes = [cls for cls, count in class_counts.items() if count >= self.min_samples_per_class]
        mask = np.isin(y, valid_classes)
        return X[mask], y[mask]

    def split_data_for_nodes(self, X: np.ndarray, y: np.ndarray, num_nodes: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Split data across nodes with stratified sampling"""
        node_data = []
        unique_classes = np.unique(y)

        for node_idx in range(num_nodes):
            indices = []
            for cls in unique_classes:
                class_indices = np.where(y == cls)[0]
                # Ensure we don't try to select more indices than available
                sample_size = max(1, min(len(class_indices)//num_nodes, len(class_indices)))
                selected = np.random.choice(class_indices, size=sample_size, replace=False)
                indices.extend(selected)
            
            np.random.shuffle(indices)
            node_X = X[indices]
            node_y = y[indices]
            node_data.append((node_X, node_y))
            logging.info(f"Node {node_idx} data: {node_X.shape}, Classes: {Counter(node_y)}")
        
        return node_data