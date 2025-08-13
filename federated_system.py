

# federated_system.py
import logging
from typing import List
from collections import Counter
from node import Node


class FederatedSystem:
    def __init__(self, num_nodes: int, input_dim: int, num_classes: int):
        self.nodes = []
        self.num_nodes = num_nodes
        self.current_server = 0
        self.round = 0

        # Initialize nodes
        for node_id in range(num_nodes):
            node = Node(node_id, num_nodes, input_dim, num_classes)
            self.nodes.append(node)

        # Setup communication channels
        self._setup_communication()
        logging.info(f"Federated system initialized with {num_nodes} nodes")

    def _setup_communication(self):
        """Create P2P communication channels between nodes"""
        for sender in self.nodes:
            for receiver in self.nodes:
                if sender.node_id != receiver.node_id:
                    sender.outgoing_queues[receiver.node_id] = receiver.incoming_queue

    def start_threads(self):
        """Start all node threads"""
        for node in self.nodes:
            node.run()

    def start_round(self):
        """Execute one federated learning round"""
        logging.info(f"\n{'='*40}\nStarting Round {self.round+1}\n{'='*40}")

        # Local training on all nodes
        for node in self.nodes:
            node.train_local()

        # Get current server for this round
        server_node = self.nodes[self.current_server]
        logging.info(f"Node {server_node.node_id} selected as server for round {self.round+1}")

        # Aggregate models using current server
        server_node.aggregate_models(self.nodes)

        # Rotate server for next round
        self.current_server = (self.current_server + 1) % self.num_nodes
        self.round += 1
        return True

    def get_system_metrics(self):
        """Get overall system metrics"""
        overall_metrics = {
            'node_accuracies': {},
            'avg_accuracy': 0,
            'total_anomalies': 0,
            'total_samples': 0,
            'attack_type_summary': {}
        }

        # Collect metrics from all nodes
        accuracies = []
        total_attack_types = Counter()
        for node in self.nodes:
            # Get evaluation metrics
            eval_metrics = node.evaluate_model()
            overall_metrics['node_accuracies'][node.node_id] = eval_metrics['accuracy']
            accuracies.append(eval_metrics['accuracy'])

            # Get attack detection metrics
            attack_metrics = node.detect_attacks()
            overall_metrics['total_anomalies'] += attack_metrics['anomalies']
            overall_metrics['total_samples'] += attack_metrics['total_samples']

            # Collect attack types
            for attack_type, count in attack_metrics['attack_types'].items():
                total_attack_types[attack_type] += count

        # Calculate average accuracy
        overall_metrics['avg_accuracy'] = sum(accuracies) / len(accuracies) if accuracies else 0

        # Add attack type summary
        overall_metrics['attack_type_summary'] = dict(total_attack_types)
        return overall_metrics