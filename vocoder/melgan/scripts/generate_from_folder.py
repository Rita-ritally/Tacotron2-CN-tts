import sys, os
sys.path.append(os.path.dirname(sys.path[0]))
from vocoder.melgan.mel2wav.interface import MelVocoder
from scipy.io.wavfile import write

from pathlib import Path
from tqdm import tqdm
import argparse
import librosa
import torch
import numpy as np



generator = None       # type: Generator
_device = None

model_path='F:/cuc/2021/Realtime-tacotron-Chinese/vocoder/saved_models/pretrained'

def load_model(weights_fpath, verbose=True):
    global generator, _device

    if verbose:
        print("Building melgan")

    if torch.cuda.is_available():
        # _model = _model.cuda()
        _device = torch.device('cuda')
    else:
        _device = torch.device('cpu')

    generator = MelVocoder(model_path)

def is_loaded():
    return generator is not None

def infer_waveform(mel, progress_callback=None):
    mel = torch.FloatTensor(mel).to(_device)
    mel = mel.unsqueeze(0)

    recons = generator.inverse(mel).squeeze().cpu().numpy()
    write('F:/cuc/2021/Realtime-tacotron-Chinese/vocoder/melgan/outputs/result.wav', 16000, recons)
    return recons


#
# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--load_path", type=Path, required=True)
#     parser.add_argument("--save_path", type=Path, required=True)
#     parser.add_argument("--wav_folder", type=Path, required=True)
#     args = parser.parse_args()
#     return args
#
#
# def main():
#     args = parse_args()
#     vocoder = MelVocoder(args.load_path)
#
#     args.save_path.mkdir(exist_ok=True, parents=True)
#
#     for i, fname in tqdm(enumerate(args.wav_folder.glob("*.wav"))):
#         wavname = fname.name
#         wav, sr = librosa.core.load(fname)
#         #获取wav音频的文件名，便于存储mel谱文件
#         name = str(wavname)
#         name = name[:-4]
#
#         mel = vocoder(torch.from_numpy(wav)[None])
#         recons = vocoder.inverse(mel).squeeze().cpu().numpy()
#
#         #将wav音频转换为mel并存为npz文件
#         path = 'F:/cuc/2021/Vocoder/melgan/mels'
#         file_name = os.path.join(path, name)
#         mel = mel.cuda().data.cpu().numpy()
#         np.savez(file_name, mel)
#
#         librosa.output.write_wav(args.save_path / wavname, recons, sr=sr)
#
#
# if __name__ == "__main__":
#     main()
