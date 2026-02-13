"""
train.py

Main training script for Structural Graph Neural Network.

This script:
1. Loads structural analysis data (graph format)
2. Normalizes features and targets
3. Trains a GNN model to predict structural responses
4. Saves the best model and training curves

Usage:
    python train.py
    python train.py --epoch_num 100 --hidden_dim 256 --lr 1e-4
    python train.py --model Structure_GCN --layer_num 5
"""
# ============================================================
# SET WORKING DIRECTORY TO SCRIPT LOCATION
# ============================================================
import os
import sys

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Change to the script directory
os.chdir(SCRIPT_DIR)

# Add script directory to Python path
sys.path.insert(0, SCRIPT_DIR)

print(f"Working directory: {os.getcwd()}")
# ============================================================


import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import numpy as np
from argparse import ArgumentParser
import time
from datetime import datetime
import os
import json

# Import from our packages
from GNN import Structure_GraphNetwork, Structure_GCN, Structure_GAT, Structure_GIN
from GNN import L1_Loss, L2_Loss
from Utils import (
    get_dataset, split_dataset, get_target_index,
    normalize_dataset, denormalize_y_linear,
    node_accuracy,
    print_space, plot_learningCurve, plot_lossCurve, visualize_graph
)


def parse_arguments():
    """Parse command-line arguments."""
    parser = ArgumentParser(description='Train Structural GNN')
    
    # Dataset arguments
    parser.add_argument('--dataset_name', type=str, default='Static_Linear_Analysis',
                        help='Name of the dataset folder')
    parser.add_argument('--whatAsNode', type=str, default='NodeAsNode',
                        help='Node representation type')
    parser.add_argument('--data_num', type=int, default=100,
                        help='Number of structures to load')
    parser.add_argument('--train_ratio', type=float, default=0.9,
                        help='Fraction of data for training')
    parser.add_argument('--normalization', type=bool, default=True,
                        help='Whether to normalize data')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='Structure_GraphNetwork',
                        choices=['Structure_GraphNetwork', 'Structure_GCN', 
                                 'Structure_GAT', 'Structure_GIN'],
                        help='Model architecture to use')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension size')
    parser.add_argument('--layer_num', type=int, default=9,
                        help='Number of GNN layers')
    parser.add_argument('--aggr', type=str, default='mean',
                        choices=['mean', 'sum', 'max'],
                        help='Aggregation method')
    parser.add_argument('--dropout_p', type=float, default=0.0,
                        help='Dropout probability')
    parser.add_argument('--gnn_act', type=bool, default=True,
                        help='Use activation functions')
    parser.add_argument('--gnn_dropout', type=bool, default=True,
                        help='Use dropout in GNN layers')
    
    # Training arguments
    parser.add_argument('--target', type=str, default='all',
                        help='Output target to train on')
    parser.add_argument('--epoch_num', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--loss_function', type=str, default='L1_Loss',
                        choices=['L1_Loss', 'L2_Loss'],
                        help='Loss function to use')
    parser.add_argument('--accuracy_threshold', type=float, default=1e-4,
                        help='Threshold for accuracy calculation')
    
    return parser.parse_args()


def get_model(args, input_dim, edge_attr_dim, device):
    """Create and return the specified model."""
    model_classes = {
        'Structure_GraphNetwork': Structure_GraphNetwork,
        'Structure_GCN': Structure_GCN,
        'Structure_GAT': Structure_GAT,
        'Structure_GIN': Structure_GIN,
    }
    
    ModelClass = model_classes[args.model]
    
    model = ModelClass(
        layer_num=args.layer_num,
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        edge_attr_dim=edge_attr_dim,
        aggr=args.aggr,
        gnn_act=args.gnn_act,
        gnn_dropout=args.gnn_dropout,
        dropout_p=args.dropout_p,
        device=device
    ).to(device)
    
    return model


def train_one_epoch(model, train_loader, optimizer, criterion, 
                    y_start, y_finish, accuracy_threshold, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_elems = 0
    
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        node_out = model(data.x, data.edge_index, data.edge_attr)
        
        # Calculate loss
        loss = criterion(
            node_out[:, y_start:y_finish],
            data.y[:, y_start:y_finish],
            accuracy_threshold
        )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate accuracy
        with torch.no_grad():
            correct, elems = node_accuracy(
                node_out[:, y_start:y_finish],
                data.y[:, y_start:y_finish],
                accuracy_threshold
            )
            total_correct += correct.item()
            total_elems += elems
    
    avg_loss = total_loss / max(total_elems, 1)
    avg_accuracy = total_correct / max(total_elems, 1)
    
    return avg_loss, avg_accuracy


def evaluate(model, loader, criterion, y_start, y_finish, 
             accuracy_threshold, device):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_elems = 0
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            
            # Forward pass
            node_out = model(data.x, data.edge_index, data.edge_attr)
            
            # Calculate loss
            loss = criterion(
                node_out[:, y_start:y_finish],
                data.y[:, y_start:y_finish],
                accuracy_threshold
            )
            total_loss += loss.item()
            
            # Calculate accuracy
            correct, elems = node_accuracy(
                node_out[:, y_start:y_finish],
                data.y[:, y_start:y_finish],
                accuracy_threshold
            )
            total_correct += correct.item()
            total_elems += elems
    
    avg_loss = total_loss / max(total_elems, 1)
    avg_accuracy = total_correct / max(total_elems, 1)
    
    return avg_loss, avg_accuracy


def main():
    """Main training function."""
    
    # ==================== SETUP ====================
    print("=" * 70)
    print("STRUCTURAL GRAPH NEURAL NETWORK - TRAINING")
    print("=" * 70)
    
    # Parse arguments
    args = parse_arguments()
    
    # Create timestamp for this run
    date_str = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Run ID: {date_str}")
    
    # Print configuration
    print("\n" + "-" * 70)
    print("Configuration:")
    print("-" * 70)
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    
    # Device setup
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda:0":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # Set random seed for reproducibility
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    
    # ==================== DATA LOADING ====================
    print("\n" + "-" * 70)
    print("Loading Data:")
    print("-" * 70)
    
    # Load dataset
    dataset = get_dataset(
        dataset_name=args.dataset_name,
        whatAsNode=args.whatAsNode,
        structure_num=args.data_num
    )
    
    if len(dataset) == 0:
        print("\nERROR: No data loaded!")
        print("Please ensure your data is in the correct location:")
        print(f"  Data/{args.dataset_name}/structure_1/structure_graph_{args.whatAsNode}.pt")
        return
    
    print(f"  Loaded {len(dataset)} structures")
    
    # ==================== NORMALIZATION ====================
    # if args.normalization:
    #     print("\n" + "-" * 70)
    #     print("Normalizing Data:")
    #     print("-" * 70)
    #     dataset, norm_dict = normalize_dataset(dataset, analysis='linear')
    #     print("  ✓ Data normalized")
        
    #     # Print normalization ranges
    #     print("\n  Normalization ranges:")
    #     for key, (min_val, max_val) in list(norm_dict.items())[:5]:
    #         if isinstance(min_val, torch.Tensor):
    #             print(f"    {key}: [{min_val.item():.4f}, {max_val.item():.4f}]")
    #         else:
    #             print(f"    {key}: [{min_val}, {max_val}]")
    #     print("    ...")
    # else:
    #     norm_dict = None

        # ==================== NORMALIZATION ====================
    if args.normalization:
        print("\n" + "-" * 70)
        print("Normalizing Data:")
        print("-" * 70)
        dataset, norm_dict = normalize_dataset(dataset, analysis='linear')
        
        # Print normalization info
        print(f"\n  Normalization info:")
        print(f"    Input (x) dimensions: {norm_dict.get('x_dim', 'N/A')}")
        print(f"    Output (y) dimensions: {norm_dict.get('y_dim', 'N/A')}")
        print(f"    Edge attr dimensions: {norm_dict.get('edge_attr_dim', 'N/A')}")
    else:
        norm_dict = None
    
    # ==================== DATA SPLITTING ====================
    print("\n" + "-" * 70)
    print("Splitting Data:")
    print("-" * 70)
    
    train_dataset, valid_dataset, _ = split_dataset(
        dataset,
        train_ratio=args.train_ratio,
        valid_ratio=1 - args.train_ratio
    )
    
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(valid_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    
    # ==================== MODEL SETUP ====================
    print("\n" + "-" * 70)
    print("Setting Up Model:")
    print("-" * 70)
    
    # Get dimensions from data
    sample_data = dataset[0]
    input_dim = sample_data.x.shape[1]
    edge_attr_dim = sample_data.edge_attr.shape[1] if sample_data.edge_attr is not None else 3
    
    print(f"  Input dimension: {input_dim}")
    print(f"  Edge attribute dimension: {edge_attr_dim}")
    
    # Create model
    model = get_model(args, input_dim, edge_attr_dim, device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model: {args.model}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Create loss function
    if args.loss_function == 'L1_Loss':
        criterion = L1_Loss()
    else:
        criterion = L2_Loss()
    
    print(f"\n  Optimizer: Adam (lr={args.lr})")
    print(f"  Loss function: {args.loss_function}")
    
    # Get target indices
    y_start, y_finish = get_target_index(args.target)
    print(f"  Target: {args.target} (columns {y_start}:{y_finish})")
    
    # ==================== CREATE RESULTS DIRECTORY ====================
    save_dir = os.path.join('Results', args.dataset_name, date_str)
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n  Results will be saved to: {save_dir}")
    
    # ==================== TRAINING ====================
    print("\n" + "-" * 70)
    print("Training:")
    print("-" * 70)
    
    # Initialize tracking arrays
    accuracy_record = np.zeros((3, args.epoch_num))  # train, valid, test
    loss_record = np.zeros((3, args.epoch_num))
    
    best_valid_accuracy = 0
    best_epoch = 0
    
    start_time = time.time()
    
    print(f"\n{'Epoch':>6} {'Train Loss':>12} {'Valid Loss':>12} {'Train Acc':>10} {'Valid Acc':>10} {'Status':>10}")
    print("-" * 70)
    
    for epoch in range(args.epoch_num):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion,
            y_start, y_finish, args.accuracy_threshold, device
        )
        
        # Evaluate
        valid_loss, valid_acc = evaluate(
            model, valid_loader, criterion,
            y_start, y_finish, args.accuracy_threshold, device
        )
        
        # Record metrics
        accuracy_record[0, epoch] = train_acc
        accuracy_record[1, epoch] = valid_acc
        loss_record[0, epoch] = train_loss
        loss_record[1, epoch] = valid_loss
        
        # Check if best model
        status = ""
        if valid_acc > best_valid_accuracy:
            best_valid_accuracy = valid_acc
            best_epoch = epoch + 1
            
            # Save best model
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pt'))
            status = "✓ BEST"
        
        # Print progress
        print(f"{epoch+1:>6} {train_loss:>12.6f} {valid_loss:>12.6f} "
              f"{train_acc:>10.4f} {valid_acc:>10.4f} {status:>10}")
    
    # ==================== TRAINING COMPLETE ====================
    training_time = time.time() - start_time
    
    print("\n" + "-" * 70)
    print("Training Complete!")
    print("-" * 70)
    print(f"  Total time: {training_time/60:.2f} minutes")
    print(f"  Best validation accuracy: {best_valid_accuracy:.4f} (epoch {best_epoch})")
    print(f"  Final validation accuracy: {accuracy_record[1, -1]:.4f}")
    
    # ==================== SAVE RESULTS ====================
    print("\n" + "-" * 70)
    print("Saving Results:")
    print("-" * 70)
    
    # Save training configuration
    config = vars(args).copy()
    config['training_time_minutes'] = training_time / 60
    config['best_valid_accuracy'] = best_valid_accuracy
    config['best_epoch'] = best_epoch
    config['final_train_accuracy'] = float(accuracy_record[0, -1])
    config['final_valid_accuracy'] = float(accuracy_record[1, -1])
    config['date'] = date_str
    
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  ✓ Saved config.json")
    
    # Save normalization dictionary
    # if norm_dict is not None:
    #     norm_dict_save = {}
    #     for key, (min_val, max_val) in norm_dict.items():
    #         if isinstance(min_val, torch.Tensor):
    #             norm_dict_save[key] = [min_val.item(), max_val.item()]
    #         else:
    #             norm_dict_save[key] = [float(min_val), float(max_val)]
        
    #     with open(os.path.join(save_dir, 'norm_dict.json'), 'w') as f:
    #         json.dump(norm_dict_save, f, indent=2)
    #     print(f"  ✓ Saved norm_dict.json")

        # Save normalization dictionary
    if norm_dict is not None:
        norm_dict_save = {}
        for key, value in norm_dict.items():
            if isinstance(value, torch.Tensor):
                norm_dict_save[key] = value.tolist()
            else:
                norm_dict_save[key] = value
        
        with open(os.path.join(save_dir, 'norm_dict.json'), 'w') as f:
            json.dump(norm_dict_save, f, indent=2)
        print(f"  ✓ Saved norm_dict.json")
    
    # Save training history
    np.save(os.path.join(save_dir, 'accuracy_record.npy'), accuracy_record)
    np.save(os.path.join(save_dir, 'loss_record.npy'), loss_record)
    print(f"  ✓ Saved training history")
    
    # Plot and save learning curves
    plot_title = f"{args.model}, {args.dataset_name}\n{date_str}, target={args.target}"
    
    plot_learningCurve(accuracy_record, save_dir, title=plot_title, target=args.target)
    plot_lossCurve(loss_record, save_dir, title=plot_title, target=args.target)
    
    # Visualize a sample graph
    try:
        sample_idx = min(5, len(dataset) - 1)
        visualize_graph(dataset[sample_idx], save_dir, name="sample_graph")
    except Exception as e:
        print(f"  ⚠ Could not visualize graph: {e}")
    
    # ==================== FINAL SUMMARY ====================
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"""
  Model:                {args.model}
  Dataset:              {args.dataset_name}
  Structures:           {len(dataset)}
  Training samples:     {len(train_dataset)}
  Validation samples:   {len(valid_dataset)}
  
  Epochs:               {args.epoch_num}
  Batch size:           {args.batch_size}
  Learning rate:        {args.lr}
  Hidden dimension:     {args.hidden_dim}
  GNN layers:           {args.layer_num}
  
  Best validation acc:  {best_valid_accuracy:.4f} (epoch {best_epoch})
  Training time:        {training_time/60:.2f} minutes
  
  Results saved to:     {save_dir}
    """)
    print("=" * 70)
    print(f"Finish time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == '__main__':
    main()