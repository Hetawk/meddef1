# ...existing code...

def apply_gradient_clipping(model, max_norm=1.0):
    """
    Apply gradient clipping to prevent exploding gradients
    
    Args:
        model: PyTorch model
        max_norm: Maximum norm for gradient clipping
    """
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

# ...existing code...
