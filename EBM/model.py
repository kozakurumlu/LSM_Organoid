import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalEBM(nn.Module):
    """
    A simple conditional Energy-Based Model (EBM) using a CNN.
    It takes a 2D raster and a class label as input and outputs a scalar energy.
    """
    def __init__(self, raster_shape=(32, 32), n_classes=3):
        """
        Initializes the layers of the neural network.
        
        Args:
            raster_shape (tuple): The (height, width) of the input rasters.
            n_classes (int): The number of distinct classes (states).
        """
        super(ConditionalEBM, self).__init__()
        self.raster_shape = raster_shape
        self.n_classes = n_classes

        # --- Convolutional Body to process the raster ---
        # We'll use a simple CNN to extract features from the 2D raster data.
        # Input channels = 1 because our rasters are binary (not RGB).
        # Output channels = 8 is a reasonable starting point for feature maps.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        # After conv1: shape will be (batch_size, 8, 32, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # After pool1: shape will be (batch_size, 8, 16, 16)
        
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        # After conv2: shape will be (batch_size, 16, 16, 16)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # After pool2: shape will be (batch_size, 16, 8, 8)

        # --- Fully Connected Head to process combined features ---
        # Calculate the flattened size after convolutions and pooling
        flattened_size = 16 * (raster_shape[0] // 4) * (raster_shape[1] // 4)
        
        # This linear layer will process the concatenated vector of raster features and class label
        self.fc1 = nn.Linear(flattened_size + n_classes, 128)
        
        # The final layer outputs a single scalar value: the energy
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x, s):
        """
        Defines the forward pass of the model.
        
        Args:
            x (torch.Tensor): A batch of rasters. 
                              Shape: (batch_size, 1, height, width)
            s (torch.Tensor): A batch of integer class labels.
                              Shape: (batch_size,)
                              
        Returns:
            torch.Tensor: A batch of scalar energy values.
                          Shape: (batch_size, 1)
        """
        # 1. Process the raster `x` through the CNN body
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        # 2. Flatten the output of the CNN
        x = x.view(x.size(0), -1) # Flatten all dimensions except batch
        
        # 3. Convert the integer state labels `s` to one-hot vectors
        s_one_hot = F.one_hot(s, num_classes=self.n_classes).float()
        
        # 4. Concatenate the raster features and the one-hot state vector
        combined = torch.cat([x, s_one_hot], dim=1)
        
        # 5. Process the combined vector through the fully connected head
        combined = F.relu(self.fc1(combined))
        energy = self.fc2(combined)
        
        return energy

# --- Example Usage (for testing the model's structure) ---
if __name__ == '__main__':
    # Create a dummy batch of data
    batch_size = 4
    dummy_rasters = torch.randn(batch_size, 1, 32, 32) # (batch, channels, height, width)
    dummy_labels = torch.tensor([0, 1, 2, 1])          # (batch,)

    # Instantiate the model
    ebm = ConditionalEBM(raster_shape=(32, 32), n_classes=3)
    
    print("Model architecture:")
    print(ebm)
    
    # Perform a forward pass
    output_energy = ebm(dummy_rasters, dummy_labels)
    
    print(f"\nInput raster shape: {dummy_rasters.shape}")
    print(f"Input label shape: {dummy_labels.shape}")
    print(f"Output energy shape: {output_energy.shape}") # Should be (batch_size, 1)
    
    # Check that the output shape is correct
    assert output_energy.shape == (batch_size, 1), "Output shape is incorrect!"
    
    print("\nModel forward pass test successful!")
