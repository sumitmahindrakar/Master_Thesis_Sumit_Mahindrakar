import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

num_nodes = 5

adjacency_matrix = np.array([
    [0, 1, 0, 1, 0],
    [1, 0, 1, 1, 0],
    [0, 1, 0, 0, 1],
    [1, 1, 0, 0, 1],
    [0, 0, 1, 1, 0],
], dtype=np.float32)

print("Adjacency Matrix")
print(adjacency_matrix)
print(f"\nShape: {adjacency_matrix.shape}")

node_feature = np.array([
    [1, 0, 1],
    [0.5, 0.5, 0.0],
    [0, 1, 0],
    [1, 1, 1],
    [0, 0, 1],
], dtype=np.float32)

print("\nNode features Matrix X:")
print(node_feature)
print(f"\nShape: {node_feature.shape} (5 nodes, 3 features each)")

labels = np.array([0, 0, 1, 1, 0])

print("\nLabels (ground truth):")
print(labels)
print("Meaning: Nodes 0,1,4 are class 0; Nodes 2,3 are class 1")

# Analyze heterophily
print("\n" + "=" * 60)
print("GRAPH STRUCTURE ANALYSIS")
print("=" * 60)
for i in range(num_nodes):
    neighbors = np.where(adjacency_matrix[i] == 1)[0]
    neighbor_labels = labels[neighbors]
    same_class = np.sum(neighbor_labels == labels[i])
    diff_class = np.sum(neighbor_labels != labels[i])
    print(f"Node {i} (class {labels[i]}): {same_class} same-class neighbors, {diff_class} different-class neighbors")


class GNNLayer:
    def __init__(self, input_dim, output_dim, use_activation=True, use_skip=False):
        """
        Args:
            use_activation: If True, apply LeakyReLU
            use_skip: If True, add skip connection (residual)
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_activation = use_activation
        self.use_skip = use_skip and (input_dim == output_dim)  # Skip only if dims match
        
        # Xavier/Glorot initialization
        scale = np.sqrt(2.0 / (input_dim + output_dim))
        self.W = np.random.randn(input_dim, output_dim) * scale
        self.b = np.zeros(output_dim)
        
        # Separate weight for self-features (important for heterophily!)
        self.W_self = np.random.randn(input_dim, output_dim) * scale
        
        self.leaky_slope = 0.1  # Increased from 0.01 for better gradient flow
        
        self.cache = {}
        activation_str = "LeakyReLU" if use_activation else "None"
        skip_str = "Yes" if self.use_skip else "No"
        print(f"Created GNN Layer: {input_dim} -> {output_dim}, Activation: {activation_str}, Skip: {skip_str}")

    def compute_normalized_adjacency(self, A):
        """Compute normalized adjacency WITHOUT self-loops (we handle self separately)"""
        n = A.shape[0]
        degree = np.sum(A, axis=1) + 1e-10  # Add small epsilon
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degree))
        A_norm = D_inv_sqrt @ A @ D_inv_sqrt
        return A_norm

    def leaky_relu(self, x):
        return np.where(x > 0, x, self.leaky_slope * x)
    
    def leaky_relu_derivative(self, x):
        return np.where(x > 0, 1.0, self.leaky_slope)

    def forward(self, H, A):
        A_norm = self.compute_normalized_adjacency(A)
        
        # Separate aggregation: neighbors + self
        H_neighbors = A_norm @ H  # Aggregate from neighbors
        H_self = H  # Self features
        
        # Transform separately and combine
        H_transformed = H_neighbors @ self.W + H_self @ self.W_self + self.b
        
        # Apply activation if not output layer
        if self.use_activation:
            H_activated = self.leaky_relu(H_transformed)
        else:
            H_activated = H_transformed
        
        # Skip connection
        if self.use_skip:
            H_new = H_activated + H  # Residual connection
        else:
            H_new = H_activated

        self.cache = {
            'H_input': H,
            'A_norm': A_norm,
            'H_neighbors': H_neighbors,
            'H_transformed': H_transformed,
            'H_activated': H_activated,
            'H_output': H_new
        }
        return H_new

    def backward(self, dL_dH_out, learning_rate=0.01):
        H_input = self.cache['H_input']
        A_norm = self.cache['A_norm']
        H_neighbors = self.cache['H_neighbors']
        H_transformed = self.cache['H_transformed']

        # Skip connection gradient
        if self.use_skip:
            dL_dH_activated = dL_dH_out
            dL_dH_skip = dL_dH_out
        else:
            dL_dH_activated = dL_dH_out
            dL_dH_skip = 0

        # Activation gradient
        if self.use_activation:
            activation_grad = self.leaky_relu_derivative(H_transformed)
            dL_dH_transformed = dL_dH_activated * activation_grad
        else:
            dL_dH_transformed = dL_dH_activated

        # Gradient computations
        dL_dW = H_neighbors.T @ dL_dH_transformed
        dL_dW_self = H_input.T @ dL_dH_transformed
        dL_db = np.sum(dL_dH_transformed, axis=0)
        
        # Gradient w.r.t. input
        dL_dH_neighbors = dL_dH_transformed @ self.W.T
        dL_dH_self = dL_dH_transformed @ self.W_self.T
        
        dL_dH_in = A_norm.T @ dL_dH_neighbors + dL_dH_self
        
        if self.use_skip:
            dL_dH_in = dL_dH_in + dL_dH_skip

        # Update parameters
        self.W -= learning_rate * dL_dW
        self.W_self -= learning_rate * dL_dW_self
        self.b -= learning_rate * dL_db

        return dL_dH_in


class SimpleGNN:
    def __init__(self, input_dim, hidden_dims, output_dim, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.layers = []

        dims = [input_dim] + hidden_dims + [output_dim]

        print("\n" + "=" * 60)
        print("BUILDING GNN MODEL")
        print("=" * 60)

        for i in range(len(dims) - 1):
            is_last_layer = (i == len(dims) - 2)
            use_activation = not is_last_layer
            use_skip = not is_last_layer and (dims[i] == dims[i+1])
            
            layer = GNNLayer(dims[i], dims[i + 1], 
                           use_activation=use_activation,
                           use_skip=use_skip)
            self.layers.append(layer)

        print(f"\nTotal layers: {len(self.layers)}")
        print(f"Learning rate: {learning_rate}")

        self.loss_history = []
        self.accuracy_history = []

    def forward(self, X, A):
        H = X

        for layer in self.layers:
            H = layer.forward(H, A)

        # Softmax
        exp_score = np.exp(H - np.max(H, axis=1, keepdims=True))
        probabilities = exp_score / np.sum(exp_score, axis=1, keepdims=True)

        return probabilities

    def compute_loss(self, probabilities, labels):
        n = len(labels)
        correct_class_probs = probabilities[np.arange(n), labels]
        loss = -np.mean(np.log(correct_class_probs + 1e-10))
        return loss

    def compute_accuracy(self, probabilities, labels):
        predictions = np.argmax(probabilities, axis=1)
        accuracy = np.mean(predictions == labels)
        return accuracy

    def backward(self, probabilities, labels):
        n = len(labels)
        
        dL_dscores = probabilities.copy()
        dL_dscores[np.arange(n), labels] -= 1
        dL_dscores /= n

        dL_dH = dL_dscores

        for layer in reversed(self.layers):
            dL_dH = layer.backward(dL_dH, self.learning_rate)

    def train_step(self, X, A, labels):
        probabilities = self.forward(X, A)
        loss = self.compute_loss(probabilities, labels)
        accuracy = self.compute_accuracy(probabilities, labels)
        self.backward(probabilities, labels)
        return loss, accuracy

    def train(self, X, A, labels, epochs=100, print_every=10):
        print("\n" + "=" * 60)
        print("TRAINING")
        print("=" * 60)

        for epoch in range(epochs):
            loss, accuracy = self.train_step(X, A, labels)

            self.loss_history.append(loss)
            self.accuracy_history.append(accuracy)

            if epoch % print_every == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:4d}/{epochs}: Loss = {loss:.4f}, Accuracy = {accuracy:.2%}")

        print("\nTraining complete!")

    def predict(self, X, A):
        probabilities = self.forward(X, A)
        predictions = np.argmax(probabilities, axis=1)
        return predictions


# Create and train model
model = SimpleGNN(
    input_dim=3,
    hidden_dims=[16, 16],  # Larger hidden layers
    output_dim=2,
    learning_rate=0.3
)

model.train(
    X=node_feature,
    A=adjacency_matrix,
    labels=labels,
    epochs=500,  # More epochs
    print_every=50
)

final_predictions = model.predict(node_feature, adjacency_matrix)
final_probs = model.forward(node_feature, adjacency_matrix)

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print("\nPer-node results:")
print("-" * 60)
for i in range(num_nodes):
    pred = final_predictions[i]
    true = labels[i]
    prob = final_probs[i]
    status = "✓ CORRECT" if pred == true else "✗ WRONG"
    confidence = max(prob[0], prob[1])
    print(f"Node {i}: True={true}, Pred={pred}, Probs=[{prob[0]:.3f}, {prob[1]:.3f}] Conf={confidence:.1%} {status}")

print("-" * 60)
print(f"\nFinal Accuracy: {model.accuracy_history[-1]:.2%}")
print(f"Final Loss: {model.loss_history[-1]:.4f}")

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(model.loss_history, 'b-', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Cross-Entropy Loss', fontsize=12)
axes[0].set_title('Training Loss Over Time', fontsize=14)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(bottom=0)

axes[1].plot(model.accuracy_history, 'g-', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].set_title('Training Accuracy Over Time', fontsize=14)
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(0, 1.1)
axes[1].axhline(y=1.0, color='r', linestyle='--', label='Perfect Accuracy')
axes[1].legend()

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
plt.show()