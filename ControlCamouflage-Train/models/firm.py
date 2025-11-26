# unet.py at the top
import torch
import torch.nn as nn
import torch.fft

class FrequencyInteractionAttentionModule(nn.Module):
    def __init__(self, in_channels, attention_channels=64):
        super().__init__()
        # a tiny convolution net to generate attention map from the magnitude (frequency domain) of image features
        self.attention_net = nn.Sequential(
            nn.Conv2d(in_channels, attention_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(attention_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        # a learnable parameter to control the intensity of fusion 
        self.fusion_gate = nn.Parameter(torch.tensor(0.0))      #default value are set to 0.0

    def forward(self, image_feature, control_feature):
        # 1. convert to frequency domain
        fft_image = torch.fft.fft2(image_feature, norm="ortho")
        fft_control = torch.fft.fft2(control_feature, norm="ortho")
        
        # 2. generate frequent attention from image features 
        # utilize the magnitude map as attention source
        fft_image_magnitude = torch.abs(fft_image)
        # using fftshift to move the low-frequent info to the center that makes it convenient for CNN to operate (you can search for what FFT feature looks like)
        fft_image_magnitude_shifted = torch.fft.fftshift(fft_image_magnitude, dim=(-2, -1))
        
        # generate the attention weight
        attention_weights_shifted = self.attention_net(fft_image_magnitude_shifted)
        
        # move the attention weight back to the original layout
        attention_weights = torch.fft.ifftshift(attention_weights_shifted, dim=(-2, -1))
        
        # 3. apply attention aggregation
        # Use attention weight to modulate the frequency control features
        fft_control_attended = fft_control * attention_weights
        
        # 4. use residual connection to fuse gate 
        # fuse the original control signal and the enhanced signal, controlled by the learnable gate
        fft_control_fused = fft_control + self.fusion_gate * (fft_control_attended - fft_control)
        
        # 5. convert back to the spatial domain
        enhanced_control_feature = torch.fft.ifft2(fft_control_fused, norm="ortho").real
        
        return enhanced_control_feature