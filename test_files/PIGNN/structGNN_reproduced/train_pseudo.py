import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import numpy as np
from argparse import ArgumentParser
import time
from datetime import datetime
import os
import sys
import json

# ============================================================
# SET WORKING DIRECTORY TO SCRIPT LOCATION
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
# ============================================================

# Import from our packages
from GNN import (
    Structure_GraphNetwork_pseudo, 
    Structure_GCN_pseudo, 
    Structure_GAT_pseudo, 
    Structure_GIN_pseudo
)
from GNN import L1_Loss, L2_Loss
from Utils import (
    get_dataset, split_dataset, get_target_index,
    normalize_dataset, denormalize_grid_num,
    node_accuracy,
    print_space, plot_learningCurve, plot_lossCurve, visualize_graph
)


def parse_arguments():
    """Parse command-line arguments."""
    parser = ArgumentParser(description='Train Structural GNN (Pseudo Models)')
    
    # Dataset arguments
    parser.add_argument('--dataset_name', type=str, default='Static_Linear_Analysis')
    parser.add_argument('--whatAsNode', type=str, default='NodeAsNode_pseudo',
                        help='Use NodeAsNode_pseudo for pseudo models')
    parser.add_argument('--data_num', type=int, default=100)
    parser.add_argument('--train_ratio', type=float, default=0.9)
    parser.add_argument('--normalization', type=bool, default=True)
    
    # Model arguments
    parser.add_argument('--model', type=str, default='Structure_GraphNetwork_pseudo',
                        choices=['Structure_GraphNetwork_pseudo', 'Structure_GCN_pseudo', 
                                 'Structure_GAT_pseudo', 'Structure_GIN_pseudo'])
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--aggr', type=str, default='mean')
    parser.add_argument('--dropout_p', type=float, default=0.0)
    parser.add_argument('--gnn_act', type=bool, default=True)
    parser.add_argument('--gnn_dropout', type=bool, default=True)
    
    # Training arguments
    parser.add_argument('--target', type=str, default='all')
    parser.add_argument('--epoch_num', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--loss_function', type=str, default='L1_Loss')
    parser.add_argument('--accuracy_threshold', type=float, default=1e-4)
    
    return parser.parse_args()


def get_layer_num_from_data(data, norm_dict):
    """
    Extract the number of layers from the structure's grid dimensions.
    
    The grid_num_y (number of floors) determines how many GNN layers to use.
    This allows the model to adapt to different building heights.
    """
    # Get grid info from first node (all nodes have same grid info)
    grid_num_info = data.x[0, :3]  # First 3 features are grid dimensions
    
    # Denormalize to get actual values
    if norm_dict is not None:
        grid_num = denormalize_grid_num(grid_num_info, norm_dict)
    else:
        grid_num = grid_num_info
    
    # grid_num_y (index 1) represents vertical dimension (number of floors)
    layer_num = int(grid_num[1].item())
    
    # Ensure at least 1 layer
    layer_num = max(1, layer_num)
    
    return layer_num


def get_model(args, input_dim, edge_attr_dim, device):
    """Create and return the specified pseudo model."""
    model_classes = {
        'Structure_GraphNetwork_pseudo': Structure_GraphNetwork_pseudo,
        'Structure_GCN_pseudo': Structure_GCN_pseudo,
        'Structure_GAT_pseudo': Structure_GAT_pseudo,
        'Structure_GIN_pseudo': Structure_GIN_pseudo,
    }
    
    ModelClass = model_classes[args.model]
    
    # Note: NO layer_num parameter for pseudo models
    model = ModelClass(
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
                    y_start, y_finish, accuracy_threshold, device, norm_dict):
    """Train for one epoch with dynamic layer count."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_elems = 0
    
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        # Get dynamic layer count from this structure
        layer_num = get_layer_num_from_data(data, norm_dict)
        
        # Forward pass with dynamic layer_num
        node_out = model(data.x, data.edge_index, data.edge_attr, layer_num)
        
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
             accuracy_threshold, device, norm_dict):
    """Evaluate model with dynamic layer count."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_elems = 0
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            
            # Get dynamic layer count
            layer_num = get_layer_num_from_data(data, norm_dict)
            
            # Forward pass with dynamic layer_num
            node_out = model(data.x, data.edge_index, data.edge_attr, layer_num)
            
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
    """Main training function for pseudo models."""
    
    print("=" * 70)
    print("STRUCTURAL GNN - PSEUDO MODEL TRAINING")
    print("(Dynamic layer count based on structure dimensions)")
    print("=" * 70)
    
    print(f"\nScript directory: {SCRIPT_DIR}")
    
    # Parse arguments
    args = parse_arguments()
    
    # Create timestamp
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
    
    # Set random seed
    torch.manual_seed(0)
    
    # ==================== DATA LOADING ====================
    print("\n" + "-" * 70)
    print("Loading Data:")
    print("-" * 70)
    
    dataset = get_dataset(
        dataset_name=args.dataset_name,
        whatAsNode=args.whatAsNode,
        structure_num=args.data_num
    )
    
    if len(dataset) == 0:
        # Try with NodeAsNode if NodeAsNode_pseudo not found
        print("  Trying NodeAsNode instead of NodeAsNode_pseudo...")
        dataset = get_dataset(
            dataset_name=args.dataset_name,
            whatAsNode='NodeAsNode',
            structure_num=args.data_num
        )
    
    if len(dataset) == 0:
        print("\nERROR: No data loaded!")
        return
    
    print(f"  Loaded {len(dataset)} structures")
    
    # Show grid dimensions from first structure
    sample = dataset[0]
    print(f"\n  Sample structure grid dimensions (from x[0, :3]):")
    print(f"    Grid X: {sample.x[0, 0].item():.0f}")
    print(f"    Grid Y: {sample.x[0, 1].item():.0f} ← (determines layer_num)")
    print(f"    Grid Z: {sample.x[0, 2].item():.0f}")
    
    # ==================== NORMALIZATION ====================
    if args.normalization:
        print("\n" + "-" * 70)
        print("Normalizing Data:")
        print("-" * 70)
        dataset, norm_dict = normalize_dataset(dataset)
    else:
        norm_dict = None
    
    # ==================== DATA SPLITTING ====================
    print("\n" + "-" * 70)
    print("Splitting Data:")
    print("-" * 70)
    
    train_dataset, valid_dataset, _ = split_dataset(
        dataset, train_ratio=args.train_ratio, valid_ratio=1 - args.train_ratio
    )
    
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(valid_dataset)}")
    
    # Create data loaders (batch_size=1 for pseudo models since layer_num varies)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
    
    # ==================== MODEL SETUP ====================
    print("\n" + "-" * 70)
    print("Setting Up Model:")
    print("-" * 70)
    
    sample_data = dataset[0]
    input_dim = sample_data.x.shape[1]
    edge_attr_dim = sample_data.edge_attr.shape[1]
    
    print(f"  Input dimension: {input_dim}")
    print(f"  Edge attribute dimension: {edge_attr_dim}")
    print(f"  Layer count: DYNAMIC (from structure)")
    
    model = get_model(args, input_dim, edge_attr_dim, device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model: {args.model}")
    print(f"  Total parameters: {total_params:,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = L1_Loss() if args.loss_function == 'L1_Loss' else L2_Loss()
    
    y_start, y_finish = get_target_index(args.target)
    print(f"  Target: {args.target} (columns {y_start}:{y_finish})")
    
    # Create results directory
    save_dir = os.path.join(SCRIPT_DIR, 'Results', args.dataset_name + '_pseudo', date_str)
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n  Results: {save_dir}")
    
    # ==================== TRAINING ====================
    print("\n" + "-" * 70)
    print("Training (Pseudo Model):")
    print("-" * 70)
    
    accuracy_record = np.zeros((3, args.epoch_num))
    loss_record = np.zeros((3, args.epoch_num))
    best_valid_accuracy = 0
    best_epoch = 0
    
    start_time = time.time()
    
    print(f"\n{'Epoch':>6} {'Train Loss':>12} {'Valid Loss':>12} {'Train Acc':>10} {'Valid Acc':>10} {'Status':>10}")
    print("-" * 70)
    
    for epoch in range(args.epoch_num):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion,
            y_start, y_finish, args.accuracy_threshold, device, norm_dict
        )
        
        valid_loss, valid_acc = evaluate(
            model, valid_loader, criterion,
            y_start, y_finish, args.accuracy_threshold, device, norm_dict
        )
        
        accuracy_record[0, epoch] = train_acc
        accuracy_record[1, epoch] = valid_acc
        loss_record[0, epoch] = train_loss
        loss_record[1, epoch] = valid_loss
        
        status = ""
        if valid_acc > best_valid_accuracy:
            best_valid_accuracy = valid_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pt'))
            status = "✓ BEST"
        
        print(f"{epoch+1:>6} {train_loss:>12.6f} {valid_loss:>12.6f} "
              f"{train_acc:>10.4f} {valid_acc:>10.4f} {status:>10}")
    
    training_time = time.time() - start_time
    
    # ==================== SAVE RESULTS ====================
    print("\n" + "-" * 70)
    print("Training Complete!")
    print("-" * 70)
    print(f"  Total time: {training_time/60:.2f} minutes")
    print(f"  Best validation accuracy: {best_valid_accuracy:.4f} (epoch {best_epoch})")
    
    # Save config
    config = vars(args).copy()
    config['training_time_minutes'] = training_time / 60
    config['best_valid_accuracy'] = float(best_valid_accuracy)
    config['best_epoch'] = best_epoch
    config['model_type'] = 'pseudo'
    config['date'] = date_str
    
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  ✓ Saved config.json")
    
    # Save training history
    np.save(os.path.join(save_dir, 'accuracy_record.npy'), accuracy_record)
    np.save(os.path.join(save_dir, 'loss_record.npy'), loss_record)
    
    # Plot curves ################
    plot_title = f"{args.model}\n{date_str}, target={args.target}"
    plot_learningCurve(accuracy_record, save_dir, title=plot_title, target=args.target)
    plot_lossCurve(loss_record, save_dir, title=plot_title, target=args.target)
    
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY (PSEUDO MODEL)")

    # Plot curves #################
    # plot_title = f"{args.model}\n{date_str}, target={args.target}"
    # plot_learningCurve(accuracy_record, save_dir, title=plot_title, target=args.target)
    # plot_lossCurve(loss_record, save_dir, title=plot_title, target=args.target)
    
    # # Visualize sample graphs with different layer counts
    # print("\n  Visualizing sample graphs:")
    # try:
    #     # Show a few structures with different sizes
    #     visualized_count = 0
    #     seen_layer_nums = set()
        
    #     for idx, data in enumerate(dataset[:20]):  # Check first 20 structures
    #         layer_num = get_layer_num_from_data(data, norm_dict)
            
    #         # Only visualize if we haven't seen this layer_num yet
    #         if layer_num not in seen_layer_nums:
    #             seen_layer_nums.add(layer_num)
    #             graph_name = f"sample_graph_layers_{layer_num}"
    #             visualize_graph(data, save_dir, name=graph_name)
    #             print(f"    ✓ {graph_name}.png (structure {idx+1}, {layer_num} GNN layers)")
    #             visualized_count += 1
                
    #             # Stop after 3 different structures
    #             if visualized_count >= 3:
    #                 break
        
    #     if visualized_count == 0:
    #         # Fallback: just visualize the first structure
    #         visualize_graph(dataset[0], save_dir, name="sample_graph")
    #         print(f"    ✓ sample_graph.png")
            
    # except Exception as e:
    #     print(f"    ⚠ Could not visualize graph: {e}")
    
    # print("\n" + "=" * 70)
    # print("TRAINING SUMMARY (PSEUDO MODEL)")

    try:
        sample_idx = min(5, len(dataset) - 1)
        visualize_graph(dataset[sample_idx], save_dir, name="sample_graph")
    except Exception as e:
        print(f"  ⚠ Could not visualize graph: {e}")

    print("=" * 70)
    print(f"""
  Model:                {args.model}
  Model type:           PSEUDO (dynamic layer count)
  Structures:           {len(dataset)}
  
  Best validation acc:  {best_valid_accuracy:.4f} (epoch {best_epoch})
  Training time:        {training_time/60:.2f} minutes
  
  Results saved to:     {save_dir}
    """)
    print("=" * 70)


if __name__ == '__main__':
    main()