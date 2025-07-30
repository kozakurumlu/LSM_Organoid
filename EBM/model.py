import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalEBM(nn.Module):
    """
    A conditional EBM adjusted for FinalSpark's 8x50 raster data.
    """
    def __init__(self, raster_shape=(8, 50), n_classes=4): # Note: Adjusted for 4 classes now
        """
        Initializes the layers of the neural network.
        
        Args:
            raster_shape (tuple): The (height, width) of the input rasters.
            n_classes (int): The number of distinct classes (states).
        """
        super(ConditionalEBM, self).__init__()
        self.raster_shape = raster_shape
        self.n_classes = n_classes

        # --- Convolutional Body to process the 8x50 raster ---
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        # After conv1: shape will be (batch_size, 8, 8, 50)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # After pool1: shape will be (batch_size, 8, 4, 25)
        
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        # After conv2: shape will be (batch_size, 16, 4, 25)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # After pool2: shape will be (batch_size, 16, 2, 12) -> 25//2 = 12

        # --- Fully Connected Head ---
        # **This calculation is the main change to adapt to the new shape**
        flattened_size = 16 * (raster_shape[0] // 4) * (raster_shape[1] // 4)
        
        self.fc1 = nn.Linear(flattened_size + n_classes, 128)
        self.fc2 = nn.Linear(128, 1) # Final output is a single energy value

    def forward(self, x, s):
        """
        Defines the forward pass of the model.
        
        Args:
            x (torch.Tensor): A batch of rasters. Shape: (batch_size, 1, 8, 50)
            s (torch.Tensor): A batch of integer class labels. Shape: (batch_size,)
                              
        Returns:
            torch.Tensor: A batch of scalar energy values. Shape: (batch_size, 1)
        """
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = x.view(x.size(0), -1)
        
        s_one_hot = F.one_hot(s, num_classes=self.n_classes).float()
        
        combined = torch.cat([x, s_one_hot], dim=1)
        
        combined = F.relu(self.fc1(combined))
        energy = self.fc2(combined)
        
        return energy

# --- Example Usage (for testing the adjusted structure) ---
if __name__ == '__main__':
    # Define the new dimensions
    RASTER_SHAPE_FINALSPARK = (8, 50)
    N_CLASSES_FINALSPARK = 4 # Baseline + 3 stimulation patterns
    
    # Create a dummy batch of data with the new dimensions
    batch_size = 4
    dummy_rasters = torch.randn(batch_size, 1, *RASTER_SHAPE_FINALSPARK)
    dummy_labels = torch.tensor([0, 1, 2, 3])

    # Instantiate the model with the new shape
    ebm = ConditionalEBM(raster_shape=RASTER_SHAPE_FINALSPARK, n_classes=N_CLASSES_FINALSPARK)
    
    print("Adjusted Model architecture:")
    print(ebm)
    
    # Perform a forward pass
    output_energy = ebm(dummy_rasters, dummy_labels)
    
    print(f"\nInput raster shape: {dummy_rasters.shape}")
    print(f"Output energy shape: {output_energy.shape}")
    
    assert output_energy.shape == (batch_size, 1), "Output shape is incorrect!"