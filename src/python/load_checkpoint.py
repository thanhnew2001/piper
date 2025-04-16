import torch
from pathlib import Path
import sys
from piper_train.vits.lightning import VitsModel

def load_checkpoint(checkpoint_path):
    # Add PosixPath to safe globals
    torch.serialization.add_safe_globals([Path])
    
    # Load checkpoint with weights_only=False
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    # Extract num_symbols and num_speakers from checkpoint if available
    num_symbols = checkpoint.get('num_symbols', 256)  # Default value if not in checkpoint
    num_speakers = checkpoint.get('num_speakers', 1)  # Default value if not in checkpoint
    
    # Create model with all required parameters
    model = VitsModel(
        hidden_channels=192,
        inter_channels=192,
        filter_channels=768,
        n_layers=6,
        n_heads=2,
        num_symbols=num_symbols,
        num_speakers=num_speakers,
        plot_save_path=None,
        show_plot=False
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['state_dict'])
    return model

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python load_checkpoint.py <checkpoint_path>")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    model = load_checkpoint(checkpoint_path)
    print("Checkpoint loaded successfully!")
