"""
Utils/plot.py

Functions for visualizing training progress and results.

Main functions:
- plot_learningCurve: Plot accuracy over epochs
- plot_lossCurve: Plot loss over epochs
- visualize_graph: Visualize a graph structure
- print_space: Helper for formatted console output
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Try to import networkx for graph visualization
try:
    import networkx as nx
    from torch_geometric.utils.convert import to_networkx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False


def print_space():
    """
    Return a formatted separator for console output.
    
    Returns:
        str: Separator string with newlines and equals signs
    
    Example:
        >>> print("Some output", end=print_space())
        Some output
        
        ====================================...====
        
    """
    return "\n" * 3 + "=" * 100 + "\n" * 3


def plot_learningCurve(accuracy_record, save_model_dir, title=None, target=None):
    """
    Plot and save the learning curve (accuracy over epochs).
    
    Shows how model accuracy improves during training.
    Useful for:
    - Monitoring training progress
    - Detecting overfitting (train acc >> valid acc)
    - Deciding when to stop training
    
    Args:
        accuracy_record: numpy array of shape (3, num_epochs)
                        Row 0: training accuracy
                        Row 1: validation accuracy
                        Row 2: test accuracy (can be zeros if not used)
        save_model_dir: Directory to save the plot
        title: Plot title (optional)
        target: Target name for filename (e.g., 'all', 'disp')
    
    Saves:
        {save_model_dir}/LearningCurve_{target}.png
    
    Example:
        >>> accuracy = np.zeros((3, 100))
        >>> accuracy[0] = train_accuracies  # Training
        >>> accuracy[1] = valid_accuracies  # Validation
        >>> plot_learningCurve(accuracy, 'Results/', target='all')
    """
    # Unpack accuracy records
    train_acc, valid_acc, test_acc = accuracy_record
    epochs = list(range(1, len(train_acc) + 1))
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Build title with best validation accuracy
    if title:
        title += '\n' + f"Best Validation Accuracy: {valid_acc.max() * 100:.1f}%"
    else:
        title = f"Learning Curve\nBest Validation Accuracy: {valid_acc.max() * 100:.1f}%"
    
    # Plot training and validation curves
    plt.plot(epochs, train_acc, 'b-', label='Train', linewidth=2)
    plt.plot(epochs, valid_acc, 'r-', label='Validation', linewidth=2)
    
    # Plot test curve if available (non-zero values)
    if test_acc[-1] != 0:
        plt.plot(epochs, test_acc, 'g-', label='Test', linewidth=2)
    
    # Mark best validation accuracy
    best_epoch = np.argmax(valid_acc) + 1
    best_acc = valid_acc.max()
    plt.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.5)
    plt.scatter([best_epoch], [best_acc], color='r', s=100, zorder=5)
    plt.annotate(f'Best: {best_acc:.3f}', xy=(best_epoch, best_acc),
                 xytext=(best_epoch + len(epochs)*0.05, best_acc - 0.05),
                 fontsize=9)
    
    # Formatting
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.ylim([-0.05, 1.05])
    plt.xlim([0, len(epochs) + 1])
    plt.title(title, fontsize=11)
    
    # Ensure directory exists
    os.makedirs(save_model_dir, exist_ok=True)
    
    # Save figure
    filename = f"LearningCurve_{target}.png" if target else "LearningCurve.png"
    filepath = os.path.join(save_model_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {filepath}")


def plot_lossCurve(loss_record, save_model_dir, title=None, target=None):
    """
    Plot and save the loss curve (loss over epochs).
    
    Shows how model loss decreases during training.
    Useful for:
    - Monitoring convergence
    - Detecting training issues
    - Comparing different configurations
    
    Args:
        loss_record: numpy array of shape (3, num_epochs)
                    Row 0: training loss
                    Row 1: validation loss
                    Row 2: test loss (can be zeros if not used)
        save_model_dir: Directory to save the plot
        title: Plot title (optional)
        target: Target name for filename
    
    Saves:
        {save_model_dir}/LossCurve_{target}.png
    """
    # Unpack loss records
    train_loss, valid_loss, test_loss = loss_record
    epochs = list(range(1, len(train_loss) + 1))
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Build title
    if title:
        title += f'\nFinal Validation Loss: {valid_loss[-1]:.4f}'
    else:
        title = f"Loss Curve\nFinal Validation Loss: {valid_loss[-1]:.4f}"
    
    # Plot training and validation curves
    plt.plot(epochs, train_loss, 'b-', label='Train', linewidth=2)
    plt.plot(epochs, valid_loss, 'r-', label='Validation', linewidth=2)
    
    # Plot test curve if available
    if test_loss[-1] != 0:
        plt.plot(epochs, test_loss, 'g-', label='Test', linewidth=2)
    
    # Mark minimum validation loss
    best_epoch = np.argmin(valid_loss) + 1
    best_loss = valid_loss.min()
    plt.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.5)
    plt.scatter([best_epoch], [best_loss], color='r', s=100, zorder=5)
    
    # Formatting
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.xlim([0, len(epochs) + 1])
    plt.title(title, fontsize=11)
    
    # Use log scale if loss varies by orders of magnitude
    if train_loss.max() / (train_loss.min() + 1e-10) > 100:
        plt.yscale('log')
    
    # Ensure directory exists
    os.makedirs(save_model_dir, exist_ok=True)
    
    # Save figure
    filename = f"LossCurve_{target}.png" if target else "LossCurve.png"
    filepath = os.path.join(save_model_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {filepath}")


def visualize_graph(data, save_model_dir, name="graph"):
    """
    Visualize a graph structure and save as image.
    
    Shows nodes and edges of the structural graph.
    
    Args:
        data: PyTorch Geometric Data object
        save_model_dir: Directory to save the plot
        name: Filename (without extension)
    
    Saves:
        {save_model_dir}/{name}.png
    
    Note: Requires networkx package
    """
    if not NETWORKX_AVAILABLE:
        print("  ⚠ networkx not available, skipping graph visualization")
        return
    
    try:
        # Convert PyG graph to NetworkX
        G = to_networkx(data, to_undirected=True)
        
        # Create figure
        plt.figure(figsize=(10, 10))
        
        # Choose layout based on graph size
        num_nodes = G.number_of_nodes()
        if num_nodes < 50:
            pos = nx.spring_layout(G, k=2, iterations=50)
        else:
            pos = nx.kamada_kawai_layout(G)
        
        # Draw graph
        nx.draw(G, pos,
                node_color='lightblue',
                node_size=300,
                edge_color='gray',
                width=1.5,
                with_labels=True,
                font_size=8,
                font_weight='bold')
        
        plt.title(f"Graph Structure ({num_nodes} nodes, {G.number_of_edges()} edges)")
        
        # Ensure directory exists
        os.makedirs(save_model_dir, exist_ok=True)
        
        # Save figure
        filepath = os.path.join(save_model_dir, f"{name}.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {filepath}")
        
    except Exception as e:
        print(f"  ⚠ Could not visualize graph: {e}")


def plot_prediction_comparison(y_pred, y_true, save_model_dir, target_name="output"):
    """
    Plot predicted vs actual values as a scatter plot.
    
    Perfect predictions would lie on the diagonal line.
    
    Args:
        y_pred: Predicted values (numpy array or tensor)
        y_true: Actual values (numpy array or tensor)
        save_model_dir: Directory to save the plot
        target_name: Name of the output for title/filename
    
    Saves:
        {save_model_dir}/Prediction_{target_name}.png
    """
    # Convert to numpy if needed
    if hasattr(y_pred, 'detach'):
        y_pred = y_pred.detach().cpu().numpy()
    if hasattr(y_true, 'detach'):
        y_true = y_true.detach().cpu().numpy()
    
    # Flatten arrays
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    
    # Create figure
    plt.figure(figsize=(8, 8))
    
    # Scatter plot
    plt.scatter(y_true, y_pred, alpha=0.5, s=10)
    
    # Add diagonal line (perfect prediction)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    
    # Calculate R² score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Formatting
    plt.xlabel("Actual Values", fontsize=12)
    plt.ylabel("Predicted Values", fontsize=12)
    plt.title(f"Prediction vs Actual: {target_name}\nR² = {r2:.4f}", fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Make axes equal
    plt.axis('equal')
    plt.xlim([min_val - 0.1 * abs(max_val - min_val), max_val + 0.1 * abs(max_val - min_val)])
    plt.ylim([min_val - 0.1 * abs(max_val - min_val), max_val + 0.1 * abs(max_val - min_val)])
    
    # Ensure directory exists
    os.makedirs(save_model_dir, exist_ok=True)
    
    # Save figure
    filepath = os.path.join(save_model_dir, f"Prediction_{target_name}.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {filepath}")


def plot_error_distribution(y_pred, y_true, save_model_dir, target_name="output"):
    """
    Plot histogram of prediction errors.
    
    Args:
        y_pred: Predicted values
        y_true: Actual values
        save_model_dir: Directory to save the plot
        target_name: Name of the output for title/filename
    
    Saves:
        {save_model_dir}/Error_{target_name}.png
    """
    # Convert to numpy if needed
    if hasattr(y_pred, 'detach'):
        y_pred = y_pred.detach().cpu().numpy()
    if hasattr(y_true, 'detach'):
        y_true = y_true.detach().cpu().numpy()
    
    # Calculate errors
    errors = (y_pred - y_true).flatten()
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Histogram
    plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    
    # Add vertical line at zero
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero error')
    
    # Statistics
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    plt.axvline(x=mean_error, color='g', linestyle='-', linewidth=2, label=f'Mean: {mean_error:.4f}')
    
    # Formatting
    plt.xlabel("Prediction Error (Predicted - Actual)", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title(f"Error Distribution: {target_name}\nMean: {mean_error:.4f}, Std: {std_error:.4f}", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Ensure directory exists
    os.makedirs(save_model_dir, exist_ok=True)
    
    # Save figure
    filepath = os.path.join(save_model_dir, f"Error_{target_name}.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {filepath}")


def moving_average(record, window=10):
    """
    Calculate moving average for smoothing curves.
    
    Args:
        record: 1D array of values
        window: Window size for averaging
    
    Returns:
        Smoothed array of same length
    """
    record = np.array(record)
    average_record = np.zeros(len(record))
    half_window = window // 2
    
    for i in range(len(record)):
        start = max(0, i - half_window)
        end = min(len(record), i + half_window + 1)
        average_record[i] = record[start:end].mean()
    
    return average_record


def plot_training_summary(accuracy_record, loss_record, save_model_dir, title="Training Summary"):
    """
    Create a combined summary plot with accuracy and loss.
    
    Args:
        accuracy_record: numpy array of shape (3, num_epochs)
        loss_record: numpy array of shape (3, num_epochs)
        save_model_dir: Directory to save the plot
        title: Plot title
    
    Saves:
        {save_model_dir}/TrainingSummary.png
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = list(range(1, len(accuracy_record[0]) + 1))
    
    # Accuracy plot
    ax1 = axes[0]
    ax1.plot(epochs, accuracy_record[0], 'b-', label='Train', linewidth=2)
    ax1.plot(epochs, accuracy_record[1], 'r-', label='Validation', linewidth=2)
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Accuracy")
    ax1.set_title(f"Accuracy (Best: {accuracy_record[1].max():.3f})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    
    # Loss plot
    ax2 = axes[1]
    ax2.plot(epochs, loss_record[0], 'b-', label='Train', linewidth=2)
    ax2.plot(epochs, loss_record[1], 'r-', label='Validation', linewidth=2)
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Loss")
    ax2.set_title(f"Loss (Final: {loss_record[1][-1]:.4f})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    # Ensure directory exists
    os.makedirs(save_model_dir, exist_ok=True)
    
    # Save figure
    filepath = os.path.join(save_model_dir, "TrainingSummary.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {filepath}")