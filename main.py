import logging
import sys
import time
from data_loader import IoMTDataLoader
from federated_system import FederatedSystem
from sklearn.model_selection import KFold
import numpy as np
import torch


def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('iomt_federated.log'),
            logging.StreamHandler()
        ]
    )
   
    # Configuration
    DATA_PATH = "iomt_dataset.csv"  # Update with your actual dataset path
    NUM_NODES = 5
    ROUNDS = 10
   
    try:
        # Load and prepare data
        logging.info("Loading dataset...")
        loader = IoMTDataLoader(DATA_PATH)
        df = loader.load_data()
        X, y, attack_labels = loader.preprocess_data(df)
        node_data = loader.split_data_for_nodes(X, y, NUM_NODES)
       
        # Initialize federated system
        logging.info("Initializing federated system...")
        system = FederatedSystem(NUM_NODES, X.shape[1], len(np.unique(y)))
       
        # Assign data to nodes
        for i, (node_X, node_y) in enumerate(node_data):
            system.nodes[i].set_data(node_X, node_y)
       
        # Training loop
        logging.info(f"Starting federated training for {ROUNDS} rounds...")
        for round_num in range(ROUNDS):
            system.start_round()
           
            # Collect and display metrics after each round
            logging.info(f"\nRound {round_num+1} Results:")
            for node in system.nodes:
                metrics = node.get_metrics()
                if metrics['val_acc']:  # Check if metrics exist
                    logging.info(f"Node {node.node_id} - "
                                f"Val Loss: {metrics['val_loss'][-1]:.4f}, "
                                f"Val Accuracy: {metrics['val_acc'][-1]:.2f}%")
               
            # Detect attacks every 5 rounds or at the end
            if (round_num + 1) % 5 == 0 or round_num == ROUNDS - 1:
                logging.info("\nAttack Detection Results:")
                for node in system.nodes:
                    attacks = node.detect_attacks()
                    logging.info(f"Node {node.node_id}: Detected {attacks['anomalies']} anomalies "
                                f"out of {attacks['total_samples']} samples")
                    
                    # Add attack type details
                    if attacks['attack_types']:
                        attack_types_str = ", ".join([f"{attack}: {count}" for attack, count in attacks['attack_types'].items()])
                        logging.info(f"- Attack Types: {attack_types_str}")
       
        # Get final system metrics
        system_metrics = system.get_system_metrics()
        
        # Print final performance summary
        logging.info("\n" + "="*50)
        logging.info("Final Performance Metrics:")
        logging.info("Node-wise Accuracy:")
        for node_id, accuracy in system_metrics['node_accuracies'].items():
            logging.info(f"Node {node_id}: {accuracy:.2f}%")
            
        logging.info(f"Average System Accuracy: {system_metrics['avg_accuracy']:.2f}%")
        logging.info(f"Total Anomalies Detected: {system_metrics['total_anomalies']}/{system_metrics['total_samples']}")
        
        if system_metrics['attack_type_summary']:
            logging.info("Attack Type Distribution:")
            for attack_type, count in system_metrics['attack_type_summary'].items():
                logging.info(f"{attack_type}: {count}")
        
        # Save final models
        logging.info("\nSaving final models...")
        for node in system.nodes:
            torch.save({
                'cdae': node.cdae.state_dict(),
                'qdnn': node.qdnn.state_dict()
            }, f"node_{node.node_id}_model.pth")
       
        logging.info("Federated training completed successfully!")
       
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()

