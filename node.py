import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
import queue
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
from sklearn.preprocessing import StandardScaler, LabelEncoder
from cryptography.fernet import Fernet
import traceback

class CDAE(nn.Module):
    """Convolutional Denoising Autoencoder for anomaly detection"""
    def __init__(self, input_dim: int, encoding_dim: int = 32):
        super(CDAE, self).__init__()
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.9),  # Increased dropout for less overfitting
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.9),
            nn.Linear(64, encoding_dim)
        )
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.9),  # Added dropout to decoder
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.9),  # Added dropout to decoder
            nn.Linear(128, input_dim),
            nn.Sigmoid()  # For normalized reconstruction
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

class QDNN(nn.Module):
    """Quantum-inspired Deep Neural Network for classification"""
    def __init__(self, input_dim: int, num_classes: int):
        super(QDNN, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(32, 128),  # Matches CDAE encoding_dim
            nn.ReLU(),
            nn.Dropout(0.7),  # Increased dropout for less overfitting
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.7),  # Increased dropout for less overfitting
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)

class DifferentialPrivacy:
    """Adds noise to model parameters for privacy protection"""
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon
        self.delta = delta
    
    def add_noise(self, parameters: np.ndarray) -> np.ndarray:
        sensitivity = 1.0  # Adjust based on data sensitivity
        noise_scale = np.sqrt(2 * np.log(1.25/self.delta)) * sensitivity / self.epsilon
        return parameters + np.random.normal(0, noise_scale, parameters.shape)

class Node:
    """Decentralized FL Node with integrated security and P2P communication"""
    def __init__(self, node_id: int, num_nodes: int, input_dim: int, num_classes: int):
        self.node_id = node_id
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Initialize models
        self.cdae = CDAE(input_dim)
        self.qdnn = QDNN(input_dim, num_classes)
        
        # Security components
        self.dp = DifferentialPrivacy()
        self.key = Fernet.generate_key()
        self.cipher = Fernet(self.key)
        
        # Data storage
        self.local_data = None
        self.local_labels = None
        self.original_labels = None
        self.data_weight = 0  # For weighted aggregation
        
        # Communication
        self.incoming_queue = queue.Queue()
        self.outgoing_queues = {}
        
        # Training state
        self.current_round = 0
        self.metrics = defaultdict(list)
        self.best_model = None
        self.is_training = False
        
        logging.info(f"Node {self.node_id} initialized")

    def run(self):
        """Background thread for handling messages"""
        while True:
            try:
                if not self.is_training:
                    self._process_messages()
            except Exception as e:
                logging.error(f"Node {self.node_id} error: {str(e)}")

    def set_data(self, data: np.ndarray, labels: np.ndarray):
        """Set local training data"""
        self.local_data = torch.FloatTensor(data)
        self.local_labels = torch.LongTensor(labels)
        self.original_labels = labels.copy()  # Keep original labels for attack analysis
        self.data_weight = len(data)
        logging.info(f"Node {self.node_id} data set: {len(data)} samples")

    def train_local(self, epochs: int = 10):
        """Local training with early stopping"""
        if self.local_data is None:
            logging.error(f"Node {self.node_id}: No data available for training")
            return
        
        self.is_training = True
        logging.info(f"Node {self.node_id} starting local training")
        
        # Reduced weight decay to help with convergence but avoid overfitting
        optimizer = torch.optim.Adam(
            list(self.cdae.parameters()) + list(self.qdnn.parameters()), 
            lr=0.001,
            weight_decay=1e-4
        )
        
        # Split data for validation (larger validation set to better detect overfitting)
        val_size = int(0.25 * len(self.local_data))
        indices = torch.randperm(len(self.local_data))
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        
        train_data = self.local_data[train_indices]
        train_labels = self.local_labels[train_indices]
        val_data = self.local_data[val_indices]
        val_labels = self.local_labels[val_indices]
        
        best_loss = float('inf')
        patience = 0
        max_patience = 3  # Reduced patience for earlier stopping
        
        for epoch in range(epochs):
            # Training step
            self.cdae.train()
            self.qdnn.train()
            
            # Add noise for denoising (increased noise for better generalization)
            noise = torch.randn_like(train_data) * 0.15
            noisy_data = train_data + noise
            
            # Forward pass
            reconstructed, encoded = self.cdae(noisy_data)
            outputs = self.qdnn(encoded)
            
            # Loss calculation with L2 regularization
            loss_cdae = nn.MSELoss()(reconstructed, train_data)
            loss_qdnn = nn.CrossEntropyLoss()(outputs, train_labels)
            total_loss = loss_cdae + 2.0 * loss_qdnn
            
            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.cdae.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.qdnn.parameters(), 1.0)
            optimizer.step()
            
            # Validation
            self.cdae.eval()
            self.qdnn.eval()
            
            with torch.no_grad():
                val_reconstructions, val_encoded = self.cdae(val_data)
                val_outputs = self.qdnn(val_encoded)
                
                val_loss_cdae = nn.MSELoss()(val_reconstructions, val_data)
                val_loss_qdnn = nn.CrossEntropyLoss()(val_outputs, val_labels)
                val_loss = val_loss_cdae + 2.0 * val_loss_qdnn
                
                preds = torch.argmax(val_outputs, dim=1)
                val_acc = (preds == val_labels).float().mean() * 100
            
            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience = 0
                self._save_best_model()
            else:
                patience += 1
                if patience >= max_patience:
                    logging.info(f"Node {self.node_id} early stopping at epoch {epoch}")
                    break
            
            # Log metrics
            self.metrics['train_loss'].append(total_loss.item())
            self.metrics['val_loss'].append(val_loss.item())
            self.metrics['val_acc'].append(val_acc.item())
            
            if epoch % 5 == 0:
                logging.info(f"Node {self.node_id} - Epoch {epoch}: "
                            f"Train Loss = {total_loss.item():.4f}, "
                            f"Val Loss = {val_loss.item():.4f}, "
                            f"Val Acc = {val_acc.item():.2f}%")
        
        # Load best model
        if self.best_model:
            self.cdae.load_state_dict(self.best_model['cdae'])
            self.qdnn.load_state_dict(self.best_model['qdnn'])
        
        self.is_training = False
        logging.info(f"Node {self.node_id} completed local training")

    def _save_best_model(self):
        """Save best model state"""
        self.best_model = {
            'cdae': self.cdae.state_dict(),
            'qdnn': self.qdnn.state_dict()
        }

    def get_metrics(self) -> Dict:
        """Return training metrics"""
        return {
            'train_loss': self.metrics['train_loss'],
            'val_loss': self.metrics['val_loss'],
            'val_acc': self.metrics['val_acc']
        }

    def _process_messages(self):
        """Handle incoming parameter updates"""
        try:
            msg = self.incoming_queue.get(timeout=0.1)
            
            if msg['type'] == 'params':
                decrypted = self._decrypt_params(msg['params'])
                noisy = self._apply_noise_to_params(decrypted)
                self._update_model(noisy)
                
        except queue.Empty:
            pass
        except Exception as e:
            logging.error(f"Node {self.node_id} message processing error: {str(e)}")

    def _update_model(self, params: Dict):
        """Update local model with received parameters"""
        try:
            # Update CDAE
            cdae_state_dict = self.cdae.state_dict()
            for key in cdae_state_dict:
                if key in params['cdae']:
                    param_tensor = torch.tensor(params['cdae'][key])
                    if param_tensor.shape == cdae_state_dict[key].shape:
                        cdae_state_dict[key] = param_tensor
            self.cdae.load_state_dict(cdae_state_dict)
            
            # Update QDNN
            qdnn_state_dict = self.qdnn.state_dict()
            for key in qdnn_state_dict:
                if key in params['qdnn']:
                    param_tensor = torch.tensor(params['qdnn'][key])
                    if param_tensor.shape == qdnn_state_dict[key].shape:
                        qdnn_state_dict[key] = param_tensor
            self.qdnn.load_state_dict(qdnn_state_dict)
            
        except Exception as e:
            logging.error(f"Node {self.node_id} model update error: {str(e)}")

    def _apply_noise_to_params(self, params: Dict) -> Dict:
        """Apply differential privacy noise to parameters"""
        noisy_params = {'cdae': {}, 'qdnn': {}}
        
        for key, value in params['cdae'].items():
            noisy_params['cdae'][key] = self.dp.add_noise(value)
            
        for key, value in params['qdnn'].items():
            noisy_params['qdnn'][key] = self.dp.add_noise(value)
            
        return noisy_params

    def get_parameters(self) -> Dict:
        """Get model parameters for sharing"""
        params = {'cdae': {}, 'qdnn': {}}
        
        # Extract CDAE parameters
        for name, param in self.cdae.named_parameters():
            params['cdae'][name] = param.data.cpu().numpy()
            
        # Extract QDNN parameters
        for name, param in self.qdnn.named_parameters():
            params['qdnn'][name] = param.data.cpu().numpy()
            
        return params

    def _encrypt_params(self, params: Dict) -> Dict:
        """Encrypt model parameters"""
        encrypted = {'cdae': {}, 'qdnn': {}}
        
        for key, value in params['cdae'].items():
            encrypted['cdae'][key] = self.cipher.encrypt(value.tobytes())
            
        for key, value in params['qdnn'].items():
            encrypted['qdnn'][key] = self.cipher.encrypt(value.tobytes())
            
        return encrypted

    def _decrypt_params(self, params: Dict) -> Dict:
        """Decrypt received parameters"""
        decrypted = {'cdae': {}, 'qdnn': {}}
        
        for key, value in params['cdae'].items():
            decrypted['cdae'][key] = np.frombuffer(self.cipher.decrypt(value))
            
        for key, value in params['qdnn'].items():
            decrypted['qdnn'][key] = np.frombuffer(self.cipher.decrypt(value))
            
        return decrypted

    def send_parameters(self, target_node_id: int):
        """Send parameters to another node"""
        if target_node_id not in self.outgoing_queues:
            logging.error(f"Node {self.node_id}: Invalid target node {target_node_id}")
            return
            
        try:
            params = self.get_parameters()
            encrypted = self._encrypt_params(params)
            
            message = {
                'type': 'params',
                'sender': self.node_id,
                'params': encrypted,
                'weight': self.data_weight
            }
            
            self.outgoing_queues[target_node_id].put(message)
            logging.info(f"Node {self.node_id} sent parameters to Node {target_node_id}")
            
        except Exception as e:
            logging.error(f"Node {self.node_id} parameter sending error: {str(e)}")

    def aggregate_models(self, nodes: List['Node']):
        """Aggregate models from all nodes (server role)"""
        logging.info(f"Node {self.node_id} acting as server for aggregation")
        
        try:
            # Collect parameters from all nodes
            all_params = []
            total_weight = 0
            
            for node in nodes:
                params = node.get_parameters()
                weight = node.data_weight
                all_params.append({
                    'params': params,
                    'weight': weight
                })
                total_weight += weight
            
            # Weighted averaging
            aggregated = {'cdae': {}, 'qdnn': {}}
            
            # CDAE parameters
            for key in all_params[0]['params']['cdae']:
                weighted_sum = np.zeros_like(all_params[0]['params']['cdae'][key])
                for p in all_params:
                    weighted_sum += p['params']['cdae'][key] * p['weight']
                aggregated['cdae'][key] = weighted_sum / total_weight
            
            # QDNN parameters
            for key in all_params[0]['params']['qdnn']:
                weighted_sum = np.zeros_like(all_params[0]['params']['qdnn'][key])
                for p in all_params:
                    weighted_sum += p['params']['qdnn'][key] * p['weight']
                aggregated['qdnn'][key] = weighted_sum / total_weight
            
            # Broadcast to all nodes
            for node in nodes:
                if node.node_id != self.node_id:
                    # Update server node directly
                    node._update_model(aggregated)
            
            # Update self
            self._update_model(aggregated)
            
            logging.info(f"Node {self.node_id} completed aggregation")
            
        except Exception as e:
            logging.error(f"Node {self.node_id} aggregation error: {str(e)}")

    def detect_attacks(self, threshold: float = 0.9) -> Dict:
        """Detect attacks using reconstruction error"""
        if self.local_data is None:
            return {'anomalies': 0, 'total_samples': 0, 'attack_types': {}}
        
        self.cdae.eval()
        
        with torch.no_grad():
            reconstructions, _ = self.cdae(self.local_data)
            errors = torch.mean((self.local_data - reconstructions)**2, dim=1)
            q = torch.quantile(errors, threshold)
            anomalies = errors > q
            
            # Calculate attack type distribution if available
            attack_types = {}
            if hasattr(self, 'original_labels') and self.original_labels is not None:
                anomaly_indices = anomalies.nonzero().squeeze().cpu().numpy()
                
                # Handle edge cases for array dimensionality
                if len(anomaly_indices.shape) == 0 and anomaly_indices.size > 0:
                    anomaly_indices = np.array([anomaly_indices.item()])
                elif len(anomaly_indices) == 0:
                    anomaly_indices = np.array([])
                
                if len(anomaly_indices) > 0:
                    # Create mock attack types for demonstration if not already defined
                    attack_names = ['SQL Injection', 'DDoS', 'Data Exfiltration', 
                                  'Ransomware', 'Malware Upload', 'Phishing']
                    
                    # Map class indices to attack names or use the original labels
                    anomaly_labels = self.original_labels[anomaly_indices]
                    
                    # Create a mapping of numerical labels to attack names for demonstration
                    attack_mapping = {}
                    for i, label in enumerate(np.unique(anomaly_labels)):
                        attack_mapping[label] = attack_names[i % len(attack_names)]
                    
                    # Count occurrences of each attack type
                    attack_counter = Counter()
                    for label in anomaly_labels:
                        if isinstance(label, (int, np.integer)):
                            attack_counter[attack_mapping.get(label, f"Unknown-{label}")] += 1
                        else:
                            attack_counter[label] += 1
                    
                    attack_types = dict(attack_counter)
            
        return {
            'total_samples': len(self.local_data),
            'anomalies': anomalies.sum().item(),
            'avg_error': errors.mean().item(),
            'attack_types': attack_types
        }
        
    def evaluate_model(self):
        """Evaluate model on all local data"""
        if self.local_data is None:
            return {'accuracy': 0, 'confusion_matrix': None}
        
        self.cdae.eval()
        self.qdnn.eval()
        
        with torch.no_grad():
            _, encodings = self.cdae(self.local_data)
            outputs = self.qdnn(encodings)
            preds = torch.argmax(outputs, dim=1)
            
            # Calculate accuracy
            correct = (preds == self.local_labels).float().sum()
            accuracy = (correct / len(self.local_data)) * 100
            
            # Generate confusion matrix
            conf_matrix = {}
            for true_label in range(self.num_classes):
                conf_matrix[true_label] = {}
                for pred_label in range(self.num_classes):
                    mask = (self.local_labels == true_label) & (preds == pred_label)
                    conf_matrix[true_label][pred_label] = mask.sum().item()
            
        return {
            'accuracy': accuracy.item(),
            'confusion_matrix': conf_matrix
        }