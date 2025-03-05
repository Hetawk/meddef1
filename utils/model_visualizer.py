import os
import sys
import argparse
import torch
from torch import nn
import numpy as np
from pathlib import Path
import logging
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, BoxStyle, Circle, Ellipse, RegularPolygon, Polygon, FancyBboxPatch
import matplotlib.colors as mcolors
import matplotlib.transforms as mtransforms
from matplotlib.lines import Line2D
import torchvision

# Try to import NetworkX, but provide fallbacks if it fails
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    logging.warning("NetworkX not installed. Some visualization features will be limited.")

# Add the project root to the path to import from project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model_loader import ModelLoader
from loader.dataset_loader import DatasetLoader  # Import DatasetLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ModelVisualizer:
    """Class for visualizing and analyzing model architectures"""
    
    def __init__(self, output_dir="out/model_visualizations"):
        """Initialize the visualizer with output directory in 'out' folder"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset_loader = DatasetLoader()  # Initialize DatasetLoader
    
    def _get_layer_info(self, module, input_size=None):
        """Extract information about a specific layer"""
        layer_info = {
            'type': module.__class__.__name__,
            'parameters': sum(p.numel() for p in module.parameters() if p.requires_grad)
        }
        
        # Add layer-specific details
        if isinstance(module, nn.Conv2d):
            layer_info.update({
                'in_channels': module.in_channels,
                'out_channels': module.out_channels,
                'kernel_size': module.kernel_size,
                'stride': module.stride,
                'padding': module.padding
            })
        elif isinstance(module, nn.Linear):
            layer_info.update({
                'in_features': module.in_features,
                'out_features': module.out_features,
                'bias': module.bias is not None
            })
        elif isinstance(module, nn.BatchNorm2d):
            layer_info.update({
                'num_features': module.num_features,
                'eps': module.eps,
                'momentum': module.momentum
            })
        elif isinstance(module, nn.MaxPool2d):
            layer_info.update({
                'kernel_size': module.kernel_size,
                'stride': module.stride,
                'padding': module.padding,
                'dilation': module.dilation
            })
        
        return layer_info
    
    def generate_model_summary(self, model, input_size=(3, 224, 224), batch_size=1):
        """Generate a detailed text summary of the model"""
        # Ensure the model is on the correct device
        model = model.to(self.device)
        
        # Create a string buffer for the summary
        summary = []
        summary.append("=" * 80)
        summary.append(f"Model: {model.__class__.__name__}")
        summary.append("=" * 80)
        
        # Get total parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        summary.append(f"Total Parameters: {total_params:,} ({trainable_params:,} trainable)")
        summary.append("-" * 80)
        
        # Headers
        summary.append(f"{'Layer (type)':<40} {'Output Shape':<20} {'Param #':<10}")
        summary.append("=" * 80)
        
        # Prepare for forward hook
        layer_details = []
        hooks = []
        
        def hook_fn(module, input, output):
            # Calculate output shape
            if isinstance(output, (list, tuple)):
                output_shape = [tuple(o.shape) for o in output]
            else:
                output_shape = tuple(output.shape)
            
            # Calculate parameters
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            layer_details.append({
                'name': module._get_name(),
                'output_shape': output_shape,
                'params': params
            })
            
        # Register hooks
        for name, module in model.named_modules():
            if not any(isinstance(module, t) for t in 
                    [nn.Sequential, nn.ModuleList, nn.ModuleDict, model.__class__]):
                hooks.append(module.register_forward_hook(hook_fn))
        
        # Execute a forward pass
        try:
            input_tensor = torch.zeros(batch_size, *input_size).to(self.device)
            model(input_tensor)
        except Exception as e:
            summary.append(f"Error running forward pass: {str(e)}")
        finally:
            # Remove hooks
            for h in hooks:
                h.remove()
        
        # Add layer details to summary
        for i, layer in enumerate(layer_details):
            output_shape_str = str(layer['output_shape']).replace("(", "[").replace(")", "]")
            summary.append(f"{i:<3} {layer['name']:<36} {output_shape_str:<20} {layer['params']:,}")
        
        # Create section for model structure
        summary.append("\n" + "=" * 80)
        summary.append("Model Structure")
        summary.append("=" * 80)
        
        # Add the model structure string
        summary.append(str(model))
        
        return "\n".join(summary)
    
    def _color_for_layer(self, layer_type):
        """Return a color based on layer type for visualization"""
        color_map = {
            'Conv2d': '#3498db',      # Blue
            'BatchNorm2d': '#2ecc71', # Green
            'ReLU': '#f1c40f',        # Yellow
            'MaxPool2d': '#e74c3c',   # Red
            'Linear': '#9b59b6',      # Purple
            'Dropout': '#95a5a6',     # Gray
            'AdaptiveAvgPool2d': '#1abc9c', # Light Blue
            'Tanh': '#d35400',        # Orange
            'Sigmoid': '#34495e',     # Dark Blue
        }
        
        return color_map.get(layer_type, '#7f8c8d')  # Default gray
    
    def _format_params(self, params):
        """Format parameter count for display"""
        if params > 1_000_000:
            return f"{params/1_000_000:.1f}M params"
        elif params > 1_000:
            return f"{params/1_000:.1f}K params"
        else:
            return f"{params} params"
    
    def _shape_for_layer(self, layer_type):
        """Return the appropriate shape for each layer type"""
        shape_map = {
            'Conv2d': 'rectangle',           # Rectangle
            'BatchNorm2d': 'rounded_rectangle', # Rounded rectangle
            'ReLU': 'triangle',              # Triangle (activation)
            'MaxPool2d': 'hexagon',          # Hexagon (pooling)
            'Linear': 'rectangle',           # Rectangle
            'Dropout': 'oval',               # Oval (regularization)
            'AdaptiveAvgPool2d': 'hexagon',  # Hexagon (pooling)
            'Softmax': 'triangle',           # Triangle (activation)
            'Sigmoid': 'triangle',           # Triangle (activation)
            'Tanh': 'triangle',              # Triangle (activation)
        }
        
        return shape_map.get(layer_type, 'rectangle')
    
    def _draw_node_shape(self, ax, shape_type, x, y, width, height, color, alpha=0.7, **kwargs):
        """Draw various shapes for nodes"""
        if shape_type == 'rectangle':
            shape = Rectangle((x - width/2, y - height/2), width, height, 
                              facecolor=color, alpha=alpha, edgecolor='black', 
                              linewidth=1.5, zorder=1, **kwargs)
            
        elif shape_type == 'rounded_rectangle':
            # Use FancyBboxPatch instead of Rectangle for rounded corners
            shape = FancyBboxPatch(
                (x - width/2, y - height/2), width, height,
                boxstyle=f"round,pad=0.2",
                facecolor=color, alpha=alpha, edgecolor='black',
                linewidth=1.5, zorder=1, **kwargs
            )
            
        elif shape_type == 'oval':
            shape = Ellipse((x, y), width, height, 
                           facecolor=color, alpha=alpha, edgecolor='black',
                           linewidth=1.5, zorder=1, **kwargs)
            
        elif shape_type == 'triangle':
            shape = RegularPolygon((x, y), 3, radius=width/1.5, 
                                 facecolor=color, alpha=alpha, edgecolor='black',
                                 linewidth=1.5, zorder=1, **kwargs)
            
        elif shape_type == 'hexagon':
            shape = RegularPolygon((x, y), 6, radius=width/1.7, 
                                  facecolor=color, alpha=alpha, edgecolor='black',
                                  linewidth=1.5, zorder=1, **kwargs)
        else:
            # Default to rectangle
            shape = Rectangle((x - width/2, y - height/2), width, height, 
                             facecolor=color, alpha=alpha, edgecolor='black', 
                             linewidth=1.5, zorder=1, **kwargs)
            
        ax.add_patch(shape)
        return shape

    def visualize_model(self, model, model_name, model_depth):
        """Create a visual representation of the model architecture with improved layout"""
        # Count layers to dynamically adjust figure size
        num_layers = sum(1 for _ in model.modules() if not any(
            isinstance(_, t) for t in [nn.Sequential, nn.ModuleList, nn.ModuleDict, model.__class__]
        ))
        
        # Adjust figure size based on model complexity
        fig_height = max(15, min(30, num_layers * 0.4))  # Scale height with layer count
        fig_width = 14  # Wider figure for better readability
        
        plt.figure(figsize=(fig_width, fig_height))
        
        # Use a horizontal layout with better spacing
        layer_height = 0.6
        vertical_spacing = 0.3  # Increased spacing between layers
        
        layers = []
        max_width = 0
        max_params = 1  # Avoid log(0) issues
        
        # Go through all modules, ignoring container modules
        for name, module in model.named_modules():
            if not any(isinstance(module, t) for t in 
                    [nn.Sequential, nn.ModuleList, nn.ModuleDict, model.__class__]):
                layer_type = module.__class__.__name__
                
                # Calculate layer width based on parameters
                params = sum(p.numel() for p in module.parameters() if p.requires_grad)
                max_params = max(max_params, params if params > 0 else 1)
                layers.append((name, layer_type, params))
        
        # Normalize layer widths based on maximum parameter count
        max_log_params = np.log10(max_params)
        for i, (name, layer_type, params) in enumerate(layers):
            # Scale width based on log of parameters, with a minimum width
            width = max(2.0, min(8.0, 2.0 + 6.0 * (np.log10(params + 1) / max_log_params))) if params > 0 else 2.0
            max_width = max(max_width, width)
            layers[i] = (name, layer_type, params, width)
        
        # Plot all layers with more space
        y_offset = 0.5  # Start with some margin at the top
        for i, (name, layer_type, params, width) in enumerate(layers):
            color = self._color_for_layer(layer_type)
            
            plt.gca().add_patch(
                Rectangle((0.5, y_offset), width, layer_height, 
                          facecolor=color, edgecolor='black', alpha=0.7)
            )
            
            # Add layer name and parameter count on the same line
            param_text = self._format_params(params)
            plt.text(0.5 + width/2, y_offset + layer_height/2, f"{layer_type} ({param_text})", 
                     ha='center', va='center', fontsize=10, fontweight='bold')
            
            y_offset += layer_height + vertical_spacing
        
        # Set plot limits with margins
        x_margin = 1.0
        y_margin = 0.5
        plt.xlim(0, max_width + x_margin)
        plt.ylim(0, y_offset + y_margin)
        
        plt.axis('off')
        plt.title(f"Model Architecture: {model_name}_{model_depth}", fontsize=16, fontweight='bold', pad=20)
        
        # Create a legend with better layout
        legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=self._color_for_layer(layer_type), 
                                        edgecolor='black', alpha=0.7, label=layer_type)
                          for layer_type in sorted(set(l[1] for l in layers))]
        
        plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.02),
                  ncol=min(5, len(legend_elements)), fontsize=10)
        
        # Add model stats with better positioning
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Format parameter counts for better readability
        if total_params > 1_000_000:
            total_params_str = f"{total_params/1_000_000:.2f}M"
            trainable_params_str = f"{trainable_params/1_000_000:.2f}M"
        else:
            total_params_str = f"{total_params/1_000:.1f}K"
            trainable_params_str = f"{trainable_params/1_000:.1f}K"
            
        plt.figtext(0.5, -0.01, f"Total Parameters: {total_params_str} ({trainable_params_str} trainable)", 
                   ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
        
        # Ensure proper layout with tight_layout (before returning figure)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        return plt.gcf()
    
    def visualize_model_structure(self, model, model_name, model_depth):
        """
        Create a structural visualization of the model architecture 
        that's better suited for papers and DrawIO recreation
        """
        # Create a container for layer information, similar structure to NetworkX
        nodes = []
        edges = []
        
        # Track layer types and their counts for unique naming
        layer_counters = {}
        ordered_layers = []
        
        # First pass: collect node information
        for name, module in model.named_modules():
            # Skip container modules
            if any(isinstance(module, t) for t in 
                   [nn.Sequential, nn.ModuleList, nn.ModuleDict, model.__class__]):
                continue
                
            layer_type = module.__class__.__name__
            
            # Keep track of how many of this layer type we've seen
            if layer_type not in layer_counters:
                layer_counters[layer_type] = 0
            layer_counters[layer_type] += 1
            
            # Create a unique node ID
            node_id = f"{layer_type}_{layer_counters[layer_type]}"
            
            # Get node properties
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            # Simplify parameter text for better fit
            if params > 1_000_000:
                param_text = f"{params/1_000_000:.1f}M"
            elif params > 1_000:
                param_text = f"{params/1_000:.1f}K"
            else:
                param_text = f"{params}"
            
            # Get shape information if available
            shape_info = ""
            if hasattr(module, "in_features") and hasattr(module, "out_features"):
                shape_info = f"\n({module.in_features}→{module.out_features})"
            elif hasattr(module, "in_channels") and hasattr(module, "out_channels"):
                shape_info = f"\n({module.in_channels}→{module.out_channels})"
                if hasattr(module, "kernel_size"):
                    # Handle different kernel_size formats
                    if isinstance(module.kernel_size, int):
                        k_size = f"{module.kernel_size}×{module.kernel_size}"
                    else:
                        k_size = "×".join(str(k) for k in module.kernel_size)
                    shape_info += f"\nk={k_size}"
            
            # Create clean label with parameter count included
            if params > 0:
                label = f"{layer_type}\n{param_text} params{shape_info}"
            else:
                label = f"{layer_type}{shape_info}"
                
            # Add node attributes
            nodes.append({
                'id': node_id, 
                'type': layer_type,
                'params': params,
                'label': label,
                'shape': self._shape_for_layer(layer_type)
            })
            ordered_layers.append(node_id)
        
        # Create edges between consecutive nodes
        for i in range(len(ordered_layers) - 1):
            edges.append((ordered_layers[i], ordered_layers[i+1]))
        
        # Setup figure - make it larger for complex models
        fig_width = 14
        # Scale height based on number of nodes
        fig_height = max(20, min(40, len(nodes) * 0.35))
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Define node positions - manual hierarchical layout for better control
        positions = {}
        
        # Group nodes by layer type to organize them
        layer_type_groups = {}
        for node in nodes:
            layer_type = node['type']
            if layer_type not in layer_type_groups:
                layer_type_groups[layer_type] = []
            layer_type_groups[layer_type].append(node)
        
        # Calculate positions in a grid layout
        # We'll place similar layer types in columns
        columns = {}
        column_assignment = {
            'Conv2d': 0,
            'BatchNorm2d': 1,
            'ReLU': 2,
            'MaxPool2d': 3,
            'Linear': 4,
            'Dropout': 5,
            'AdaptiveAvgPool2d': 3,  # Same column as MaxPool
            'Softmax': 2,            # Same column as ReLU
            'Sigmoid': 2,            # Same column as ReLU
            'Tanh': 2,               # Same column as ReLU
        }
        
        # Place nodes in a flow-chart like structure
        vertical_spacing = 3.0  # More spacing between nodes
        x_spacing = 4.0        # Spacing between columns
        
        # First, assign each node to its position based on original order
        for i, node in enumerate(nodes):
            node_id = node['id']
            # Use the original order for y-position to maintain flow
            y_pos = -i * vertical_spacing
            
            # Assign x position based on layer type if possible
            layer_type = node['type']
            x_pos = column_assignment.get(layer_type, i % 5) * x_spacing + 2
            
            positions[node_id] = (x_pos, y_pos)
        
        # Define node colors by layer type
        layer_types = set(node['type'] for node in nodes)
        colors = plt.cm.tab20(np.linspace(0, 1, len(layer_types)))
        color_map = {layer: colors[i] for i, layer in enumerate(layer_types)}
        
        # Draw nodes with custom styling and different shapes
        for node in nodes:
            layer_type = node['type']
            label = node['label']
            node_id = node['id']
            shape_type = node['shape']
            
            x, y = positions[node_id]
            
            # Scale shape dimensions based on label length
            text_lines = len(label.split('\n'))
            width = 2.0 + 0.2 * max(len(line) for line in label.split('\n'))
            height = 1.0 + 0.2 * text_lines
            
            # Draw appropriate shape for this layer type
            color = color_map[layer_type]
            self._draw_node_shape(ax, shape_type, x, y, width, height, color)
            
            # Add layer text with appropriate size
            fontsize = min(10, 14 - 0.4 * text_lines)  # Reduce font size for long labels
            ax.text(x, y, label, ha='center', va='center', 
                   fontsize=fontsize, fontweight='bold', zorder=2)
        
        # Draw edges with arrows and better routing
        arrow_props = dict(
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.2",
            mutation_scale=15,
            linewidth=1.5,
            color='black',
            zorder=0
        )
        
        for src, dst in edges:
            x1, y1 = positions[src]
            x2, y2 = positions[dst]
            
            # Calculate arrow start/end points to avoid overlapping with nodes
            src_node = next(n for n in nodes if n['id'] == src)
            dst_node = next(n for n in nodes if n['id'] == dst)
            
            # Create arrow with custom styling
            arrow = FancyArrowPatch(
                (x1, y1 - 0.8),  # Bottom of source
                (x2, y2 + 0.8),  # Top of destination
                **arrow_props
            )
            ax.add_patch(arrow)
        
        # Add legend for layer types
        legend_elements = [Line2D([0], [0], marker='s', color='w', 
                                 markerfacecolor=color_map[layer_type], 
                                 markersize=10, label=layer_type)
                          for layer_type in sorted(layer_types)]
        
        plt.legend(handles=legend_elements, loc='upper center',
                  bbox_to_anchor=(0.5, 0.02), ncol=min(5, len(legend_elements)))
        
        # Set plot limits with extra padding to ensure all elements are visible
        all_xs = [pos[0] for pos in positions.values()]
        all_ys = [pos[1] for pos in positions.values()]
        
        if all_xs and all_ys:  # Ensure there are values
            x_min, x_max = min(all_xs), max(all_xs)
            y_min, y_max = min(all_ys), max(all_ys)
            x_range = max(x_max - x_min, 1) 
            y_range = max(y_max - y_min, 1)
            
            # Add extra padding for larger models
            x_padding = x_range * 0.25
            y_padding = y_range * 0.15
            
            ax.set_xlim(x_min - x_padding, x_max + x_padding)
            ax.set_ylim(y_min - y_padding, y_max + y_padding)
        else:
            # Default limits
            ax.set_xlim(-5, 20)
            ax.set_ylim(-40, 5)
        
        # Add title
        ax.set_title(f"Model Structure: {model_name}_{model_depth}", fontsize=16, fontweight='bold', pad=20)
        
        # Hide axis
        ax.axis('off')
        
        # Add model stats
        total_params = sum(p.numel() for p in model.parameters())
        
        # Format parameter counts
        if total_params > 1_000_000:
            params_text = f"Total Params: {total_params/1_000_000:.2f}M"
        else:
            params_text = f"Total Params: {total_params/1_000:.1f}K"
            
        plt.figtext(0.5, 0.01, params_text, ha='center', fontsize=12, 
                   bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.98])  # Adjust layout to make room for text
        return fig

    def visualize_flowchart(self, model, model_name, model_depth):
        """
        Create an enhanced flowchart-style visualization specifically for papers
        with improved spacing and multiple shapes
        """
        # Group layers by types to simplify the visualization
        layer_groups = []
        current_group = None
        
        # Function to identify layer category
        def get_layer_category(layer_type):
            categories = {
                'conv': ['Conv2d'],
                'norm': ['BatchNorm2d', 'LayerNorm', 'InstanceNorm2d', 'GroupNorm'],
                'activation': ['ReLU', 'Sigmoid', 'Tanh', 'LeakyReLU', 'PReLU', 'ELU', 'GELU'],
                'pooling': ['MaxPool2d', 'AvgPool2d', 'AdaptiveAvgPool2d', 'AdaptiveMaxPool2d'],
                'dropout': ['Dropout', 'Dropout2d'],
                'linear': ['Linear'],
                'attention': ['SelfAttention'],
                'defense': ['DefenseModule', 'AdversarialFeatureDetector', 'MedicalFeatureExtractor', 'MultiScaleFeatures'],
                'other': []
            }
            
            for category, types in categories.items():
                if layer_type in types:
                    return category
            return 'other'
        
        # First, collect all layers
        layers = []
        for name, module in model.named_modules():
            if not any(isinstance(module, t) for t in 
                    [nn.Sequential, nn.ModuleList, nn.ModuleDict, model.__class__]):
                layer_type = module.__class__.__name__
                params = sum(p.numel() for p in module.parameters() if p.requires_grad)
                
                # Get shape information if available
                shape_info = None
                if hasattr(module, "in_features") and hasattr(module, "out_features"):
                    shape_info = f"{module.in_features}→{module.out_features}"
                elif hasattr(module, "in_channels") and hasattr(module, "out_channels"):
                    shape_info = f"{module.in_channels}→{module.out_channels}"
                
                # Add category
                category = get_layer_category(layer_type)
                
                layers.append({
                    'type': layer_type,
                    'params': params,
                    'shape_info': shape_info,
                    'category': category,
                    'name': name
                })
        
        # For a complex model, we'll group consecutive layers of the same category
        # This is now more aggressive to reduce image size
        groups = []
        current_group = None
        
        # More aggressive grouping - only create a new group for major category changes
        major_categories = {
            'input': ['conv', 'norm'], 
            'processing': ['activation', 'pooling', 'dropout'],
            'feature': ['defense', 'attention'],
            'output': ['linear']
        }
        
        def get_major_category(category):
            for major, categories in major_categories.items():
                if category in categories:
                    return major
            return 'other'
        
        for layer in layers:
            major_cat = get_major_category(layer['category'])
            
            if current_group is None or current_group['major_category'] != major_cat:
                if current_group:
                    groups.append(current_group)
                
                current_group = {
                    'category': layer['category'],
                    'major_category': major_cat,
                    'layers': [layer],
                    'params': layer['params'],
                    'types': [layer['type']]
                }
            else:
                current_group['layers'].append(layer)
                current_group['params'] += layer['params']
                if layer['type'] not in current_group['types']:
                    current_group['types'].append(layer['type'])
        
        # Add the last group
        if current_group:
            groups.append(current_group)
        
        # Count blocks and limit figure size
        num_blocks = len(groups)
        logging.info(f"Created {num_blocks} block groups for flowchart visualization")
        
        # Use a fixed height per group, with a reasonable maximum
        fixed_height_per_group = 1.5
        fig_height = min(40, max(8, num_blocks * fixed_height_per_group))
        fig_width = 10  # Fixed width
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Define box properties with more compact spacing
        box_width = 3.5
        box_height = 1.0
        box_spacing = 1.0  # Reduced spacing between boxes
        
        # Define colors based on layer categories
        category_colors = {
            'conv': '#4285F4',      # Google Blue
            'norm': '#34A853',      # Google Green
            'activation': '#FBBC05', # Google Yellow
            'pooling': '#EA4335',   # Google Red
            'dropout': '#7F8C8D',   # Gray
            'linear': '#9C27B0',    # Purple
            'attention': '#FF9800',  # Orange
            'defense': '#00ACC1',    # Cyan
            'other': '#607D8B'      # Blue Gray
        }
        
        # Define shapes for different categories
        category_shapes = {
            'conv': 'rectangle',
            'norm': 'rounded_rectangle',
            'activation': 'triangle',
            'pooling': 'hexagon',
            'dropout': 'oval',
            'linear': 'rectangle',
            'attention': 'hexagon',
            'defense': 'rounded_rectangle',
            'other': 'rectangle'
        }
        
        # Draw grouped boxes and arrows
        y_positions = []
        
        total_params = sum(p.numel() for p in model.parameters())
        
        # Calculate max y position based on figure height
        max_y_pos = fig_height - 1
        step_size = min((fig_height - 4) / (len(groups) + 1), box_spacing + box_height)
        
        for i, group in enumerate(groups):
            # Calculate y position (vertically stacked)
            y_pos = max_y_pos - i * step_size
            y_positions.append(y_pos)
            
            # Format parameter text
            params = group['params']
            percentage = (params / total_params) * 100
            
            if params > 1_000_000:
                param_text = f"{params/1_000_000:.1f}M ({percentage:.1f}%)"
            elif params > 1_000:
                param_text = f"{params/1_000:.1f}K ({percentage:.1f}%)"
            else:
                param_text = f"{params} ({percentage:.1f}%)"
            
            # Create label text - show representative info
            if len(group['types']) > 3:
                types_label = f"{len(group['types'])} layers"
            else:
                types_label = ", ".join(set(group['types']))
                if len(types_label) > 25:  # Truncate if too long
                    types_label = types_label[:22] + "..."
            
            # Include count of layers if multiple
            count_suffix = f" (×{len(group['layers'])})" if len(group['layers']) > 1 else ""
            label = f"{types_label}{count_suffix}\n{param_text}"
            
            # Add shape info only from representative layers
            if group['layers'][0]['shape_info'] and len(label.split('\n')) < 3:
                label += f"\n{group['layers'][0]['shape_info']}"
            
            # Get color and shape based on category - use first layer's category
            color = category_colors.get(group['category'], '#607D8B')
            shape_type = category_shapes.get(group['category'], 'rectangle')
            
            # Draw the shape
            self._draw_node_shape(ax, shape_type, fig_width/2, y_pos, 
                                box_width, box_height, color, alpha=0.8)
            
            # Add text inside the box
            ax.text(fig_width/2, y_pos, label,
                  ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Draw arrows connecting boxes
        for i in range(len(y_positions) - 1):
            arrow_start_y = y_positions[i] - box_height/2
            arrow_end_y = y_positions[i+1] + box_height/2
            
            # Create arrow with custom styling
            arrow = FancyArrowPatch(
                (fig_width/2, arrow_start_y),
                (fig_width/2, arrow_end_y),
                arrowstyle="-|>",
                connectionstyle="arc3,rad=0.0",
                mutation_scale=15,
                linewidth=1.5,
                color='black',
                zorder=0
            )
            ax.add_patch(arrow)
        
        # Add title
        plt.title(f"Model Architecture: {model_name}", fontsize=16, fontweight='bold', pad=20)
        
        # Remove axes
        plt.axis('off')
        
        # Add model stats at bottom
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Format parameter counts for better readability
        if total_params > 1_000_000:
            total_params_str = f"{total_params/1_000_000:.2f}M"
            trainable_str = f"{trainable_params/1_000_000:.2f}M"
        else:
            total_params_str = f"{total_params/1_000:.1f}K"
            trainable_str = f"{trainable_params/1_000:.1f}K"
            
        plt.figtext(0.5, 0.01, 
                   f"Total Parameters: {total_params_str} ({trainable_str} trainable)",
                   ha="center", fontsize=12, 
                   bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
        
        # Simplify legend - create by major categories instead of all categories
        major_cat_colors = {
            'input': '#4285F4',      # Blue
            'processing': '#34A853',  # Green
            'feature': '#FBBC05',     # Yellow
            'output': '#9C27B0',      # Purple
            'other': '#607D8B'        # Gray
        }
        
        legend_elements = []
        for major_cat, color in major_cat_colors.items():
            element = Rectangle((0, 0), 1, 0.5, facecolor=color, 
                               edgecolor='black', alpha=0.7, label=major_cat.capitalize())
            legend_elements.append(element)
        
        ax.legend(handles=legend_elements, loc='upper center', 
                bbox_to_anchor=(0.5, 1.02), ncol=min(5, len(legend_elements)))
        
        # Expand the axis limits slightly to ensure all elements are visible
        ax.set_xlim(0, fig_width)
        ax.set_ylim(0, fig_height)
        
        # Use manual tight layout settings
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.90)
        
        return fig

    def get_num_classes(self, dataset_name):
        """Determine the number of classes in the dataset"""
        try:
            # Try to load dataset to get class information
            train_loader, _, _ = self.dataset_loader.load_data(
                dataset_name=dataset_name,
                batch_size={'train': 1, 'val': 1, 'test': 1},
                num_workers=0,
                pin_memory=False
            )
            
            # Get number of classes from dataset
            dataset = train_loader.dataset
            if hasattr(dataset, 'classes'):
                num_classes = len(dataset.classes)
                logging.info(f"Found {num_classes} classes in dataset {dataset_name}: {dataset.classes}")
                return num_classes
            elif hasattr(dataset, 'class_to_idx'):
                num_classes = len(dataset.class_to_idx)
                logging.info(f"Found {num_classes} classes in dataset {dataset_name}")
                return num_classes
            else:
                logging.warning(f"Could not determine number of classes from dataset {dataset_name}. Using default: 4")
                return 4
                
        except Exception as e:
            logging.error(f"Error loading dataset {dataset_name}: {str(e)}")
            logging.warning("Using default number of classes: 4")
            return 4

    def analyze_model(self, arch, depth, input_channels=3, dataset_name=None, num_classes=None):
        """Analyze a model given architecture and depth"""
        try:
            # Determine number of classes if not provided
            if num_classes is None and dataset_name is not None:
                num_classes = self.get_num_classes(dataset_name)
            elif num_classes is None:
                logging.warning("No dataset or num_classes specified. Using default: 4")
                num_classes = 4
            
            logging.info(f"Using {num_classes} output classes for model visualization")
            
            # Convert depth to appropriate format
            if isinstance(depth, str) and depth.replace('.', '', 1).isdigit():
                depth = float(depth)
            
            # Create depth dictionary for ModelLoader
            depth_dict = {arch: [depth]}
            
            # Initialize model loader
            model_loader = ModelLoader(self.device, [arch])
            
            # Load model
            models_and_names = model_loader.get_model(
                model_name=arch,
                depth=depth_dict,
                input_channels=input_channels,
                num_classes=num_classes,
                task_name="visualization",
                dataset_name=dataset_name if dataset_name else "dummy"
            )
            
            # Extract model and model name
            if not models_and_names:
                raise ValueError(f"No model found for architecture {arch} with depth {depth}")
            
            model, model_name = models_and_names[0]
            
            # Generate output directory for this model - now inside out/model_visualizations
            model_output_dir = os.path.join(self.output_dir, f"{model_name}")
            os.makedirs(model_output_dir, exist_ok=True)
            
            # Generate model summary
            summary = self.generate_model_summary(model)
            summary_path = os.path.join(model_output_dir, f"{model_name}_summary.txt")
            with open(summary_path, 'w') as f:
                f.write(summary)
            
            # Create visualization with improved layout
            fig = self.visualize_model(model, arch, depth)
            vis_path = os.path.join(model_output_dir, f"{model_name}_visualization.png")
            fig.savefig(vis_path, bbox_inches='tight', dpi=150)  # Higher DPI for better quality
            plt.close(fig)
            
            # Create new structural visualization for papers/DrawIO
            struct_vis_path = None
            struct_svg_path = None
            flowchart_path = None
            flowchart_svg_path = None
            
            try:
                struct_fig = self.visualize_model_structure(model, arch, depth)
                struct_vis_path = os.path.join(model_output_dir, f"{model_name}_structure.png")
                struct_fig.savefig(struct_vis_path, bbox_inches='tight', dpi=150)  # Reduced DPI from 300 to 150
                
                # Also save as SVG for vector graphics in papers
                struct_svg_path = os.path.join(model_output_dir, f"{model_name}_structure.svg")
                struct_fig.savefig(struct_svg_path, bbox_inches='tight', format='svg')
                plt.close(struct_fig)
                
                logging.info(f"Structural visualization saved to: {struct_vis_path}")
                logging.info(f"SVG version saved to: {struct_svg_path}")
            except Exception as e:
                logging.error(f"Error generating structural visualization: {str(e)}")
                import traceback
                logging.error(traceback.format_exc())
            
            # Create flowchart-style visualization for papers
            try:
                flowchart_fig = self.visualize_flowchart(model, arch, depth)
                flowchart_path = os.path.join(model_output_dir, f"{model_name}_flowchart.png")
                flowchart_fig.savefig(flowchart_path, bbox_inches='tight', dpi=150)  # Reduced DPI from 300 to 150
                
                # Also save as SVG
                flowchart_svg_path = os.path.join(model_output_dir, f"{model_name}_flowchart.svg")
                flowchart_fig.savefig(flowchart_svg_path, bbox_inches='tight', format='svg')
                plt.close(flowchart_fig)
                
                logging.info(f"Flowchart visualization saved to: {flowchart_path}")
            except Exception as e:
                logging.error(f"Error generating flowchart visualization: {str(e)}")
                import traceback
                logging.error(traceback.format_exc())
            
            logging.info(f"Model analysis complete!")
            logging.info(f"Summary saved to: {summary_path}")
            logging.info(f"Visualization saved to: {vis_path}")
            
            return {
                'model': model,
                'model_name': model_name,
                'summary_path': summary_path,
                'visualization_path': vis_path,
                'structure_path': struct_vis_path,
                'structure_svg_path': struct_svg_path,
                'flowchart_path': flowchart_path,
                'flowchart_svg_path': flowchart_svg_path,
                'num_classes': num_classes
            }
            
        except Exception as e:
            logging.error(f"Error analyzing model: {str(e)}")
            raise


def parse_args():
    parser = argparse.ArgumentParser(description='Model Visualization Tool')
    parser.add_argument('--arch', required=True, type=str, help='Model architecture (e.g., meddef1_)')
    parser.add_argument('--depth', required=True, type=str, help='Model depth (e.g., 1.0)')
    parser.add_argument('--input_channels', type=int, default=3, help='Number of input channels')
    parser.add_argument('--output_dir', type=str, default='out/model_visualizations', 
                      help='Directory to save visualizations')
    parser.add_argument('--dataset', type=str, default=None, 
                      help='Dataset name to auto-detect number of classes')
    parser.add_argument('--num_classes', type=int, default=None,
                      help='Optional: Manually specify number of classes (overrides dataset detection)')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    visualizer = ModelVisualizer(output_dir=args.output_dir)
    result = visualizer.analyze_model(
        arch=args.arch,
        depth=args.depth,
        input_channels=args.input_channels,
        dataset_name=args.dataset,
        num_classes=args.num_classes
    )
    
    print(f"\nModel visualization complete!")
    print(f"Model created with {result['num_classes']} output classes")
    print(f"Summary: {result['summary_path']}")
    print(f"Detailed visualization: {result['visualization_path']}")
    print(f"Network structure visualization: {result.get('structure_path', 'Not generated')}")
    print(f"Flowchart visualization: {result.get('flowchart_path', 'Not generated')}")

