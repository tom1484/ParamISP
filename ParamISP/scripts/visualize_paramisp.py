#!/usr/bin/env python
import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np

# Import modules directly since we're now inside the ParamISP folder
import utils.io
import utils.convert
import utils.metrics
import data.utils
import data.modules
import models.paramisp
import layers.bayer
import layers.color
from models.paramisp import CommonArgs
from models.paramisp import ParamISP

class ParamISPVisualizer:
    """Visualizes the intermediate steps of ParamISP forward and inverse processes."""
    
    def __init__(self, forward_model_path=None, inverse_model_path=None, output_dir="visualization_output", device="cuda"):
        """
        Initialize the visualizer.
        
        Args:
            forward_model_path: Path to the pretrained forward model checkpoint
            inverse_model_path: Path to the pretrained inverse model checkpoint (if None, uses forward_model_path)
            output_dir: Directory to save visualization results
            device: Device to run the model on
        """
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories for better organization
        self.stages = {
            "raw": self.output_dir / "01_raw",
            "mosaic_wb": self.output_dir / "02_mosaic_wb",
            "wb_cm": self.output_dir / "03_wb_cm",
            "cm_localnet": self.output_dir / "04_cm_localnet",
            "localnet_tonenet": self.output_dir / "05_localnet_tonenet",
            "rgb": self.output_dir / "06_rgb",
            "cycle": self.output_dir / "07_cycle",
            "inv_cycle": self.output_dir / "08_inv_cycle",
        }
        
        # Create all directories
        for dir_path in self.stages.values():
            dir_path.mkdir(exist_ok=True, parents=True)
        
        # Load forward model if path is provided
        if forward_model_path is not None:
            print(f"Loading forward model weights from {forward_model_path}")
            self.forward_args = CommonArgs(inverse=False)
            self.forward_model = ParamISP(self.forward_args)
            weights = utils.io.loadpt(forward_model_path)["state_dict"]
            self.forward_model.load_state_dict(weights)
            self.forward_model = self.forward_model.to(device)
            self.forward_model.eval()
        else:
            self.forward_model = None
        
        # Load inverse model if path is provided
        if inverse_model_path is not None:
            print(f"Loading inverse model weights from {inverse_model_path}")
            self.inverse_args = CommonArgs(inverse=True)
            self.inverse_model = ParamISP(self.inverse_args)
            weights = utils.io.loadpt(inverse_model_path)["state_dict"]
            self.inverse_model.load_state_dict(weights)
            self.inverse_model = self.inverse_model.to(device)
            self.inverse_model.eval()
        else:
            self.inverse_model = None
    
    def save_tensor_as_image(self, tensor, stage, name, is_raw=False):
        """
        Save a tensor as an image.
        
        Args:
            tensor: The tensor to save (should already be on CPU and detached)
            name: Base name for the saved file
            stage: Processing stage (used for directory)
            is_raw: Whether this is a RAW image
        """
        # Get the appropriate directory
        output_dir = self.stages.get(stage, self.output_dir)
        
        # Create full path with appropriate extension
        extension = ".tiff" if is_raw else ".png"
        filename = f"{name}{extension}"
        
        # Save as appropriate format
        if is_raw:
            utils.io.savetiff(tensor, output_dir, filename, u16=True)
        else:
            utils.io.saveimg(tensor, output_dir, filename)
    
    def visualize_forward_process(self, batch):
        """Visualize the forward process (RAW to RGB)."""
        if self.forward_model is None:
            print("No forward model loaded. Skipping forward process visualization.")
            return None
            
        print("Visualizing forward process (RAW to RGB)...")
        
        # Store original forward method
        original_forward = self.forward_model.forward
        
        # Create hooks and intermediate results storage
        intermediates = {}
        
        # Define a wrapper for the forward method to capture intermediates
        def forward_wrapper(self, batch, training_mode=False, extra=False):
            # Get embedding
            embed = self.get_embedding(batch, training_mode)
            # intermediates['embedding'] = embed.clone()
            
            # Start with RAW
            x = batch["raw"]
            intermediates['raw'] = x.detach().cpu()
            
            # Demosaic
            x = self.demosaic(x, batch["bayer_pattern"])
            intermediates['mosaic_wb'] = x.detach().cpu()
            
            # White balance
            x = layers.bayer.apply_white_balance(x, batch["bayer_pattern"], batch["white_balance"], mosaic_flag=False)
            intermediates['wb_cm'] = x.detach().cpu()
            
            # Clip values
            x = x.clip(0, 1)
            
            # Color matrix
            x = layers.color.apply_color_matrix(x, batch["color_matrix"])
            intermediates['cm_localnet'] = x.detach().cpu()
            
            # Get common features
            common_features = self.get_common_features(x, batch)
            # intermediates['common_features'] = common_features.clone()
            
            # Process through local network
            z = self.get_input_features(x, common_features)
            x = self.hyperlocalnet(x, z, embed)
            intermediates['localnet_tonenet'] = x.detach().cpu()
            
            # Process through tone network
            z = self.get_input_features(x, common_features)
            x = self.hypertonenet(x, z, embed)
            intermediates['rgb'] = x.detach().cpu()
            
            return x
        
        # Replace the forward method temporarily
        self.forward_model.forward = forward_wrapper.__get__(self.forward_model, ParamISP)
        
        # Process the batch
        with torch.no_grad():
            output = self.forward_model(batch)
        
        # Save intermediate results
        for name, tensor in intermediates.items():
            is_raw = name == 'raw'
            self.save_tensor_as_image(tensor, name, "forward", is_raw=is_raw)
        
        # Restore original forward method
        self.forward_model.forward = original_forward
        
        return output
    
    def visualize_inverse_process(self, batch):
        """Visualize the inverse process (RGB to RAW)."""
        if self.inverse_model is None:
            print("No inverse model loaded. Skipping inverse process visualization.")
            return None
            
        print("Visualizing inverse process (RGB to RAW)...")
        
        # Store original forward method
        original_forward = self.inverse_model.forward
        
        # Create hooks and intermediate results storage
        intermediates = {}
        
        # Define a wrapper for the forward method to capture intermediates
        def inverse_wrapper(self: ParamISP, batch, training_mode=False, extra=False):
            # Get embedding
            embed = self.get_embedding(batch, training_mode)
            # intermediates['embedding'] = embed.clone()
            
            # Start with RGB
            x = batch["rgb"]
            intermediates['rgb'] = x.detach().cpu()
            
            # Get common features
            common_features = self.get_common_features(x, batch)
            # intermediates['common_features'] = common_features.clone()
            
            # Process through tone network (first in inverse process)
            z = self.get_input_features(x, common_features)
            x = self.hypertonenet(x, z, embed)
            intermediates['localnet_tonenet'] = x.detach().cpu()
            
            # Process through local network (second in inverse process)
            z = self.get_input_features(x, common_features)
            x = self.hyperlocalnet(x, z, embed)
            intermediates['cm_localnet'] = x.detach().cpu()
            
            # Inverse color matrix
            x = layers.color.apply_color_matrix(x, batch["color_matrix"].inverse())
            intermediates['wb_cm'] = x.detach().cpu()
            
            # Inverse white balance
            x = layers.bayer.apply_white_balance(x, batch["bayer_pattern"], 1/batch["white_balance"], mosaic_flag=False)
            intermediates['mosaic_wb'] = x.detach().cpu()
            
            # Mosaic
            x = layers.bayer.mosaic(x, batch["bayer_pattern"])
            intermediates['raw'] = x.detach().cpu()
            
            return x
        
        # Replace the forward method temporarily
        self.inverse_model.forward = inverse_wrapper.__get__(self.inverse_model, ParamISP)
        
        # Process the batch
        with torch.no_grad():
            output = self.inverse_model(batch)
        
        # Save intermediate results
        for stage, tensor in intermediates.items():
            is_raw = stage == 'raw'
            self.save_tensor_as_image(tensor, stage, "inverse", is_raw=is_raw)
        
        # Restore original forward method
        self.inverse_model.forward = original_forward
        
        return output
    
    def visualize_full_cycle(self, batch):
        """Visualize the full cycle: RAW → RGB → RAW."""
        if self.forward_model is None or self.inverse_model is None:
            print("Both forward and inverse models must be loaded for full cycle visualization.")
            return None, None
            
        print("Visualizing full cycle (RAW → RGB → RAW)...")
        
        # Forward process
        with torch.no_grad():
            rgb_output = self.forward_model(batch)
            # Ensure RGB output is in 0-1 range
            rgb_output = rgb_output.clip(0, 1)
        
        # Create a new batch with the RGB output
        inverse_batch = {
            "rgb": rgb_output,
            "raw": batch["raw"],
            "bayer_pattern": batch["bayer_pattern"],
            "white_balance": batch["white_balance"],
            "color_matrix": batch["color_matrix"],
            "focal_length": batch["focal_length"],
            "f_number": batch["f_number"],
            "exposure_time": batch["exposure_time"],
            "iso_sensitivity": batch["iso_sensitivity"],
            "quantized_level": batch["quantized_level"],
            "camera_name": batch["camera_name"]
        }
        
        # Inverse process
        with torch.no_grad():
            raw_output = self.inverse_model(inverse_batch)
            # Ensure RAW output is in 0-1 range
            raw_output = raw_output.clip(0, 1)
        
        # Save the results
        self.save_tensor_as_image(batch["raw"], "cycle", "original_raw", is_raw=True)
        self.save_tensor_as_image(rgb_output, "cycle", "rgb_output")
        self.save_tensor_as_image(raw_output, "cycle", "reconstructed_raw", is_raw=True)
        
        # Calculate error between original and reconstructed RAW
        error = torch.abs(batch["raw"] - raw_output)
        error_normalized = error / (error.max() + 1e-8)  # Add small epsilon to avoid division by zero
        self.save_tensor_as_image(error_normalized, "cycle", "raw_error")
        
        # Calculate metrics
        psnr = utils.metrics.psnr(raw_output.cpu(), batch["raw"].cpu(), 1.0)
        ssim = utils.metrics.ssim(raw_output.cpu(), batch["raw"].cpu())
        
        # Save metrics to a text file
        metrics_file = self.stages["cycle"] / "metrics.txt"
        with open(metrics_file, 'w') as f:
            f.write(f"RAW → RGB → RAW Metrics:\n")
            f.write(f"  PSNR: {psnr:.2f} dB\n")
            f.write(f"  SSIM: {ssim:.4f}\n")
        
        print(f"RAW → RGB → RAW Metrics:")
        print(f"  PSNR: {psnr:.2f} dB")
        print(f"  SSIM: {ssim:.4f}")
        
        return rgb_output, raw_output
    
    def visualize_inverse_cycle(self, batch):
        """Visualize the inverse cycle: RGB → RAW → RGB."""
        if self.forward_model is None or self.inverse_model is None:
            print("Both forward and inverse models must be loaded for inverse cycle visualization.")
            return None, None
            
        print("Visualizing inverse cycle (RGB → RAW → RGB)...")
        
        # Inverse process (RGB to RAW)
        with torch.no_grad():
            raw_output = self.inverse_model(batch)
            # Ensure RAW output is in 0-1 range
            raw_output = raw_output.clip(0, 1)
        
        # Create a new batch with the RAW output
        forward_batch = {
            "raw": raw_output,
            "rgb": batch["rgb"],
            "bayer_pattern": batch["bayer_pattern"],
            "white_balance": batch["white_balance"],
            "color_matrix": batch["color_matrix"],
            "focal_length": batch["focal_length"],
            "f_number": batch["f_number"],
            "exposure_time": batch["exposure_time"],
            "iso_sensitivity": batch["iso_sensitivity"],
            "quantized_level": batch["quantized_level"],
            "camera_name": batch["camera_name"]
        }
        
        # Forward process (RAW to RGB)
        with torch.no_grad():
            rgb_reconstructed = self.forward_model(forward_batch)
            # Ensure RGB output is in 0-1 range
            rgb_reconstructed = rgb_reconstructed.clip(0, 1)
        
        # Save the results
        self.save_tensor_as_image(batch["rgb"], "inv_cycle", "original_rgb")
        self.save_tensor_as_image(raw_output, "inv_cycle", "raw_output", is_raw=True)
        self.save_tensor_as_image(rgb_reconstructed, "inv_cycle", "reconstructed_rgb")
        
        # Calculate error between original and reconstructed RGB
        error = torch.abs(batch["rgb"] - rgb_reconstructed)
        error_normalized = error / (error.max() + 1e-8)  # Add small epsilon to avoid division by zero
        self.save_tensor_as_image(error_normalized, "inv_cycle", "rgb_error")
        
        # Calculate metrics
        psnr = utils.metrics.psnr(rgb_reconstructed.cpu(), batch["rgb"].cpu(), 1.0)
        ssim = utils.metrics.ssim(rgb_reconstructed.cpu(), batch["rgb"].cpu())
        
        # Save metrics to a text file
        metrics_file = self.stages["inv_cycle"] / "metrics.txt"
        with open(metrics_file, 'w') as f:
            f.write(f"RGB → RAW → RGB Metrics:\n")
            f.write(f"  PSNR: {psnr:.2f} dB\n")
            f.write(f"  SSIM: {ssim:.4f}\n")
        
        print(f"RGB → RAW → RGB Metrics:")
        print(f"  PSNR: {psnr:.2f} dB")
        print(f"  SSIM: {ssim:.4f}")
        
        return raw_output, rgb_reconstructed

def main():
    parser = argparse.ArgumentParser(description="Visualize ParamISP forward and inverse processes")
    parser.add_argument("--forward-model", type=str, default=None, help="Path to forward model checkpoint")
    parser.add_argument("--inverse-model", type=str, default=None, help="Path to inverse model checkpoint")
    parser.add_argument("--camera", type=str, default="D7000", help="Camera model")
    parser.add_argument("--output", type=str, default="visualization_output", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to run on")
    parser.add_argument("--index", type=int, default=0, help="Image index to use")
    parser.add_argument("--forward", action="store_true", default=False, help="Run forward process visualization")
    parser.add_argument("--inverse", action="store_true", default=False, help="Run inverse process visualization")
    parser.add_argument("--cycle", action="store_true", default=False, help="Run forward cycle visualization (RAW → RGB → RAW)")
    parser.add_argument("--inv-cycle", action="store_true", default=False, help="Run inverse cycle visualization (RGB → RAW → RGB)")
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = ParamISPVisualizer(args.forward_model, args.inverse_model, args.output, args.device)
    
    # Load a sample image
    print(f"Loading sample image from camera {args.camera}, index {args.index}")
    datamodule = data.modules.CameraTestingData(
        args.camera, crop_type="full", bayer_pattern="rggb", use_extra=True,
        crop_size=1024, batch_size=1, num_workers=0, select_index=[args.index]
    )
    datamodule.setup()
    batch = next(iter(datamodule.test_dataloader()))
    
    # Move batch to device
    batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v 
             for k, v in batch.items()}
    
    # Determine which processes to run based on provided model paths
    run_forward = args.forward and args.forward_model is not None
    run_inverse = args.inverse and args.inverse_model is not None

    cycle_available = args.forward_model is not None and args.inverse_model is not None
    run_cycle = args.cycle and cycle_available
    run_inv_cycle = args.inv_cycle and cycle_available
    
    # Visualize forward process
    if run_forward:
        visualizer.visualize_forward_process(batch)
    
    # Visualize inverse process
    if run_inverse:
        visualizer.visualize_inverse_process(batch)
    
    # Visualize full cycle (RAW → RGB → RAW)
    if run_cycle:
        visualizer.visualize_full_cycle(batch)
    
    # Visualize inverse cycle (RGB → RAW → RGB)
    if run_inv_cycle:
        visualizer.visualize_inverse_cycle(batch)
    
    print(f"Visualization completed. Results saved to {args.output}/")
    print(f"Output structure:")
    print(f"  01_raw/               - RAW image")
    print(f"  02_wb_mosaic/         - Between white balance and mosaic")
    print(f"  03_cm_wb/             - Between color matrix and white balance")
    print(f"  04_localnet_cm/       - Between local network and color matrix")
    print(f"  05_tonenet_localnet/  - Between tone network and local network")
    print(f"  06_rgb/               - RGB image")
    print(f"  07_cycle/             - Forward cycle visualization (RAW → RGB → RAW)")
    print(f"  08_inv_cycle/         - Inverse cycle visualization (RGB → RAW → RGB)")

if __name__ == "__main__":
    main() 