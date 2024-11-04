# test_msarnet_with_robust_method.py
import unittest
import torch
from model.attention.MSARNet import MSARNet


class TestMSARNetWithRobustMethod(unittest.TestCase):
    def test_msarnet_with_robust_method(self):
        # Configure MSARNet with dummy robust method and attention types
        robust_method_params = {
            'method_type': 'attention',
            'input_dim': 512,
            'output_dim': 4,  # Set to match the model's num_classes
            'value_dim': 512,
            'attention_types': ['spatial', 'self']  # Include both attention types
        }
        model = MSARNet(depth=18, num_classes=4, robust_method_params=robust_method_params)

        # Create a dummy input tensor
        input_tensor = torch.randn(1, 3, 224, 224)  # Batch size of 1, 3 channels, 224x224 image

        # Forward pass
        try:
            output = model(input_tensor)
            print(f"Output shape: {output.shape}")
        except ValueError as e:
            print(f"Error caught during model forward pass: {str(e)}")

        # Expected shape is flattened after attention layers, so adapt expected shape accordingly
        expected_shape = (1, 512)  # After flattening the final output
        self.assertEqual(output.shape, expected_shape, f"Unexpected output shape: {output.shape}")


if __name__ == '__main__':
    unittest.main()
