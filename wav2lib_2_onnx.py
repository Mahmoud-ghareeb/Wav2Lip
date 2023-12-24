from os import listdir, path
import os
import argparse
import torch
from models import Wav2Lip
from torch.ao.quantization import QConfig, MinMaxObserver, default_qconfig

parser = argparse.ArgumentParser(
    description='Inference code to lip-sync videos in the wild using Wav2Lip models')

parser.add_argument('--checkpoint_path', type=str,
                    help='Path of saved checkpoint to load weights from', required=True)

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))


def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_model(path):
    model = Wav2Lip()
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()


def main():
    model = load_model(args.checkpoint_path)
    model.eval()
    
    rand_img = torch.rand(1, 6, 96, 96)
    rand_mel = torch.rand(1, 1, 80, 16)

    torch.onnx.export(model, 
                  (rand_mel, rand_img), 
                  'wav2lip.onnx', 
                  export_params=True, 
                  opset_version=11,  # Choose an appropriate opset version
                  do_constant_folding=True, 
                  input_names=['image_input', 'mel_input'], 
                  output_names=['output'],
                  dynamic_axes={'image_input': {0: 'batch_size'},  # if you have dynamic batch size
                                'mel_input': {0: 'batch_size'},
                                'output': {0: 'batch_size'}})

    print('The model has been converted to onnx')

if __name__ == '__main__':
    main()
