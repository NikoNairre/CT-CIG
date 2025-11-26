# unet.py at the top
import torch
import torch.nn as nn
import torch.fft

class FrequencyInteractionAttentionModule(nn.Module):
    def __init__(self, in_channels, attention_channels=64):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Conv2d(in_channels, attention_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(attention_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.fusion_gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, image_feature, control_feature):
        fft_image = torch.fft.fft2(image_feature, norm="ortho")
        fft_control = torch.fft.fft2(control_feature, norm="ortho")
        
        fft_image_magnitude = torch.abs(fft_image)
        fft_image_magnitude_shifted = torch.fft.fftshift(fft_image_magnitude, dim=(-2, -1))
        
        attention_weights_shifted = self.attention_net(fft_image_magnitude_shifted)
        
        attention_weights = torch.fft.ifftshift(attention_weights_shifted, dim=(-2, -1))
        
        fft_control_attended = fft_control * attention_weights
        
        fft_control_fused = fft_control + self.fusion_gate * (fft_control_attended - fft_control)
        
        enhanced_control_feature = torch.fft.ifft2(fft_control_fused, norm="ortho").real
        
        return enhanced_control_feature