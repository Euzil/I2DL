"""Models for facial keypoint detection"""

import torch
import torch.nn as nn

class KeypointModel(nn.Module):
    """Facial keypoint detection model"""
    def __init__(self, hparams):
        """
        Initialize your model from a given dict containing all your hparams
        Warning: Don't change the method declaration (i.e. by adding more
            arguments), otherwise it might not work on the submission server
            
        """
        super().__init__()
        self.hparams = hparams
        
        ########################################################################
        # TODO: Define all the layers of your CNN, the only requirements are:  #
        # 1. The network takes in a batch of images of shape (Nx1x96x96)       #
        # 2. It ends with a linear layer that represents the keypoints.        #
        # Thus, the output layer needs to have shape (Nx30),                   #
        # with 2 values representing each of the 15 keypoint (x, y) pairs      #
        #                                                                      #
        # Some layers you might consider including:                            #
        # maxpooling layers, multiple conv layers, fully-connected layers,     #
        # and other layers (such as dropout or batch normalization) to avoid   #
        # overfitting.                                                         #
        #                                                                      #
        # We would truly recommend to make your code generic, such as you      #
        # automate the calculation of the number of parameters at each layer.  #
        # You're going probably try different architecutres, and that will     #
        # allow you to be quick and flexible.                                  #
        ########################################################################
        

        # Get hyperparameters with slightly increased channels
        n_channels = [32, 64, 96, 128]  # Increased channel sizes but still manageable
        kernel_sizes = hparams.get("kernel_sizes", [3, 3, 3, 3])
        dropout_rate = hparams.get("dropout_rate", 0.2)
        use_batch_norm = hparams.get("use_batch_norm", True)
        
        layers = []
        in_channels = 1  # Input is grayscale
        
        # Create convolutional blocks with residual connections
        for i, (out_channels, kernel_size) in enumerate(zip(n_channels, kernel_sizes)):
            # Use stride=2 for first two conv layers
            stride = 2 if i < 2 else 1
            
            # Main convolution block
            layers.extend([
                # First conv layer in the block
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2, stride=stride),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity(),
                nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity(),
                
                # Second conv layer in the block (deeper feature extraction)
                nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size//2, stride=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity(),
                
                # Pooling for later layers
                nn.MaxPool2d(2, 2) if i >= 2 else nn.Identity()
            ])
            
            if dropout_rate > 0:
                layers.append(nn.Dropout2d(dropout_rate))
            
            in_channels = out_channels
        
        self.features = nn.Sequential(*layers)
        
        # Calculate flattened features size
        self.flat_features = n_channels[-1] * 6 * 6
        
        # Enhanced classifier with intermediate size
        self.classifier = nn.Sequential(
            nn.Linear(self.flat_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 30)
        )
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
    
                    
    def forward(self, x):
        
        # check dimensions to use show_keypoint_predictions later
        if x.dim() == 3:
            x = torch.unsqueeze(x, 0)
        ########################################################################
        # TODO: Define the forward pass behavior of your model                 #
        # for an input image x, forward(x) should return the                   #
        # corresponding predicted keypoints.                                   #
        # NOTE: what is the required output size?                              #
        ########################################################################


        # Feature extraction
        x = self.features(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return x


class DummyKeypointModel(nn.Module):
    """Dummy model always predicting the keypoints of the first train sample"""
    def __init__(self):
        super().__init__()
        self.prediction = torch.tensor([[
            0.4685, -0.2319,
            -0.4253, -0.1953,
            0.2908, -0.2214,
            0.5992, -0.2214,
            -0.2685, -0.2109,
            -0.5873, -0.1900,
            0.1967, -0.3827,
            0.7656, -0.4295,
            -0.2035, -0.3758,
            -0.7389, -0.3573,
            0.0086, 0.2333,
            0.4163, 0.6620,
            -0.3521, 0.6985,
            0.0138, 0.6045,
            0.0190, 0.9076,
        ]])

    def forward(self, x):
        return self.prediction.repeat(x.size()[0], 1, 1, 1)
