

def get_model_params(model):
    """Calculate the total number of parameters in the model in millions"""
    return sum(p.numel() for p in model.parameters()) / 1000000.0







