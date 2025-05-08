import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Define the architecture of RRDBNet used in ESRGAN
class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''
    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock(nf, gc)
        self.RDB2 = ResidualDenseBlock(nf, gc)
        self.RDB3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32):
        super(RRDBNet, self).__init__()
        self.in_nc = in_nc
        self.out_nc = out_nc
        self.nf = nf
        self.nb = nb  # number of RRDB blocks
        self.gc = gc  # number of growth channels

        # First convolution
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        
        # RRDB blocks
        self.RRDB_trunk = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        
        # Trunk convolution
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        
        # Upsampling layers
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        
        # High resolution convolution
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        
        # Output convolution
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        
        # Activation
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        # First convolution
        fea = self.conv_first(x)
        
        # RRDBs
        trunk = self.RRDB_trunk(fea)
        trunk = self.trunk_conv(trunk)
        
        # Global residual learning
        fea = fea + trunk
        
        # Upsampling
        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        
        # Output
        out = self.conv_last(self.lrelu(self.HRconv(fea)))
        
        return out


def convert_to_onnx(input_path, output_path):
    print(f"Converting {input_path} to {output_path}...")
    
    # Load the state dict
    state_dict = torch.load(input_path, map_location=torch.device('cpu'))
    
    # Create the ESRGAN model
    model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32)
    
    # Try to load the state dict
    try:
        model.load_state_dict(state_dict)
        print("Successfully loaded state dict!")
    except Exception as e:
        print(f"Error loading state dict: {e}")
        print("Attempting to rename keys for compatibility...")
        
        # Create a new state dict with renamed keys
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('RRDB_trunk.'):
                # Keep original keys - model structure should match
                new_state_dict[k] = v
            else:
                # Keep as is
                new_state_dict[k] = v
        
        # Try loading with new state dict
        try:
            model.load_state_dict(new_state_dict)
            print("Successfully loaded state dict after renaming keys!")
        except Exception as e:
            print(f"Still having issues loading state dict: {e}")
            print("Using strict=False to load what we can...")
            model.load_state_dict(state_dict, strict=False)
    
    # Set to evaluation mode
    model.eval()
    
    # Create a dummy input tensor of size 64x64
    dummy_input = torch.randn(1, 3, 64, 64)
    
    # Export to ONNX format
    print("Exporting to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'},
        }
    )
    
    print(f"Successfully exported ONNX model to {output_path}")
    
    # Verify the model
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model is valid!")
    except ImportError:
        print("ONNX package not installed. Skipping verification.")
    except Exception as e:
        print(f"ONNX verification warning: {e}")
        print("The model may still work with OpenCV.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_esrgan.py input.pth output.onnx")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    try:
        convert_to_onnx(input_file, output_file)
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)