RMSNorm: Replaces LayerNorm. It is computationally simpler (no mean centering) and numerically stable.
Pre-Norm: Normalization is applied before attention/FFN layers (improves gradient flow) rather than after.
SwiGLU: An activation function that replaces ReLU/GeLU. It adds a gating mechanism, increasing parameters slightly but improving convergence significantly.
Bias-Free Layers: Removing bias terms from Linear layers to improve stability.