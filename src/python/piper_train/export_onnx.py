#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
from typing import Optional

import torch

from .vits.lightning import VitsModel

_LOGGER = logging.getLogger("piper_train.export_onnx")

OPSET_VERSION = 15


def main() -> None:
    """Main entry point"""
    torch.manual_seed(1234)

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to model checkpoint (.ckpt)")
    parser.add_argument("output", help="Path to output model (.onnx)")

    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to the console"
    )
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    _LOGGER.debug(args)

    # -------------------------------------------------------------------------

    args.checkpoint = Path(args.checkpoint)
    args.output = Path(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Add PosixPath to safe globals for checkpoint loading
    torch.serialization.add_safe_globals([Path])
    
    # Load checkpoint with weights_only=False
    checkpoint = torch.load(args.checkpoint, weights_only=False)
    
    # Get number of speakers from checkpoint
    state_dict = checkpoint['state_dict']
    num_speakers = state_dict['model_g.emb_g.weight'].shape[0]
    _LOGGER.info("Number of speakers in checkpoint: %d", num_speakers)
    
    # Create model with required parameters
    model = VitsModel(
        num_symbols=256,  # Will be updated from checkpoint
        num_speakers=num_speakers,  # Use actual number of speakers
        sample_rate=22050,
        dataset=None,
        hidden_channels=192,
        inter_channels=192,
        filter_channels=768,
        n_layers=6,
        n_heads=2,
        plot_save_path=None,
        show_plot=False
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['state_dict'])
    
    model_g = model.model_g

    num_symbols = model_g.n_vocab
    num_speakers = model_g.n_speakers

    # Inference only
    model_g.eval()

    with torch.no_grad():
        model_g.dec.remove_weight_norm()

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_g.to(device)

    def infer_forward(text, text_lengths, scales, sid):
        noise_scale = scales[0]
        length_scale = scales[1]
        noise_scale_w = scales[2]
        audio = model_g.infer(
            text,
            text_lengths,
            noise_scale=noise_scale,
            length_scale=length_scale,
            noise_scale_w=noise_scale_w,
            sid=sid
        )[0].unsqueeze(1)

        return audio

    model_g.forward = infer_forward

    dummy_input_length = 50
    sequences = torch.randint(
        low=0, high=num_symbols, size=(1, dummy_input_length), dtype=torch.long
    ).to(device)
    sequence_lengths = torch.LongTensor([sequences.size(1)]).to(device)

    # Speaker ID
    sid = torch.LongTensor([0]).to(device)

    # noise, noise_w, length
    scales = torch.FloatTensor([0.667, 1.0, 0.8]).to(device)
    dummy_input = (sequences, sequence_lengths, scales, sid)

    # Export
    torch.onnx.export(
        model=model_g,
        args=dummy_input,
        f=str(args.output),
        verbose=False,
        opset_version=OPSET_VERSION,
        input_names=["input", "input_lengths", "scales", "sid"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 1: "phonemes"},
            "input_lengths": {0: "batch_size"},
            "sid": {0: "batch_size"},
            "output": {0: "batch_size", 1: "time"},
        },
    )

    _LOGGER.info("Exported model to %s", args.output)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
