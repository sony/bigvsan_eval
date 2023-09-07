import argparse
import auraloss
import functools
import json
import librosa
import numpy as np
import os
import pysptk
import torch
import torchaudio as ta

from cargan.evaluate.objective.metrics import Pitch
from cargan.preprocess.pitch import from_audio
from fastdtw import fastdtw
from pesq import pesq
from scipy.io.wavfile import read
from scipy.spatial.distance import euclidean
from tqdm import tqdm

SR_TARGET = 24000
MAX_WAV_VALUE = 32768.0


def load_wav(full_path):
    sampling_rate, audio = read(full_path)
    if sampling_rate != SR_TARGET:
        raise IOError(
            f'Sampling rate of the file {full_path} is {sampling_rate} Hz, but the model requires {SR_TARGET} Hz'
        )

    audio = audio / MAX_WAV_VALUE

    audio = torch.FloatTensor(audio)
    audio = audio.unsqueeze(0)

    return audio


def readmgc(x):
    frame_length = 1024
    hop_length = 256
    # Windowing
    frames = librosa.util.frame(x, frame_length=frame_length, hop_length=hop_length).astype(np.float64).T
    frames *= pysptk.blackman(frame_length)
    assert frames.shape[1] == frame_length
    # Order of mel-cepstrum
    order = 25
    alpha = 0.41
    stage = 5
    gamma = -1.0 / stage

    mgc = pysptk.mgcep(frames, order, alpha, gamma)
    mgc = mgc.reshape(-1, order + 1)
    return mgc


def evaluate(gt_dir, synth_dir):
    """Perform objective evaluation"""
    files = [file for file in os.listdir(synth_dir) if file.endswith('.wav')]
    gpu = 0 if torch.cuda.is_available() else None
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')
    torch.cuda.empty_cache()

    mrstft_tot = 0.0
    pesq_tot = 0.0
    s = 0.0
    frames_tot = 0

    resampler_16k = ta.transforms.Resample(SR_TARGET, 16000).to(device)
    resampler_22k = ta.transforms.Resample(SR_TARGET, 22050).to(device)

    # Modules for evaluation metrics
    loss_mrstft = auraloss.freq.MultiResolutionSTFTLoss(device=device)
    batch_metrics_periodicity = Pitch()
    periodicity_fn = functools.partial(from_audio, gpu=gpu)

    with torch.no_grad():

        iterator = tqdm(files, dynamic_ncols=True, desc=f'Evaluating {synth_dir}')
        for wavID in iterator:

            y = load_wav(os.path.join(gt_dir, wavID))
            y_g_hat = load_wav(os.path.join(synth_dir, wavID))
            y = y.to(device)
            y_g_hat = y_g_hat.to(device)

            y_16k = resampler_16k(y)
            y_g_hat_16k = resampler_16k(y_g_hat)

            y_22k = resampler_22k(y)
            y_g_hat_22k = resampler_22k(y_g_hat)

            # MRSTFT calculation
            mrstft_tot += loss_mrstft(y_g_hat, y).item()

            # PESQ calculation
            y_int_16k = (y_16k[0] * MAX_WAV_VALUE).short().cpu().numpy()
            y_g_hat_int_16k = (y_g_hat_16k[0] * MAX_WAV_VALUE).short().cpu().numpy()
            pesq_tot += pesq(16000, y_int_16k, y_g_hat_int_16k, 'wb')

            # MCD calculation
            y_double_22k = (y_22k[0] * MAX_WAV_VALUE).double().cpu().numpy()
            y_g_hat_double_22k = (y_g_hat_22k[0] * MAX_WAV_VALUE).double().cpu().numpy()

            y_mgc = readmgc(y_double_22k)
            y_g_hat_mgc = readmgc(y_g_hat_double_22k)

            _, path = fastdtw(y_mgc, y_g_hat_mgc, dist=euclidean)

            y_path = list(map(lambda l: l[0], path))
            y_g_hat_path = list(map(lambda l: l[1], path))
            y_mgc = y_mgc[y_path]
            y_g_hat_mgc = y_g_hat_mgc[y_g_hat_path]

            frames_tot += y_mgc.shape[0]

            z = y_mgc - y_g_hat_mgc
            s += np.sqrt((z * z).sum(-1)).sum()

            # Periodicity calculation
            true_pitch, true_periodicity = periodicity_fn(y_22k)
            pred_pitch, pred_periodicity = periodicity_fn(y_g_hat_22k)
            batch_metrics_periodicity.update(true_pitch, true_periodicity, pred_pitch, pred_periodicity)

    results = batch_metrics_periodicity()

    return {
        'M-STFT': mrstft_tot / len(files),
        'PESQ': pesq_tot / len(files),
        'MCD': 10.0 / np.log(10.0) * np.sqrt(2.0) * float(s) / float(frames_tot),
        'Periodicity': results['periodicity'],
        'V/UV F1': results['f1'],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('list_wavs_dir', nargs='+')
    parser.add_argument('--output_file', default=None)
    a = parser.parse_args()

    if len(a.list_wavs_dir) & 1:
        raise ValueError('The number of directories should be even.')

    # Check directories
    list_gt_dir = []
    list_synth_dir = []
    for i in range(0, len(a.list_wavs_dir), 2):
        gt_dir = a.list_wavs_dir[i]
        synth_dir = a.list_wavs_dir[i + 1]

        gt_files = set(os.listdir(gt_dir))
        synth_files = set([file for file in os.listdir(synth_dir) if file.endswith('.wav')])
        if gt_files < synth_files:
            raise IOError(
                f'Each file in "{synth_dir}" needs to have the corresponding file that has the same name in "{gt_dir}"'
            )

        list_gt_dir.append(gt_dir)
        list_synth_dir.append(synth_dir)

    # Evaluate waveforms
    results_tot = {
        'M-STFT': 0.0,
        'PESQ': 0.0,
        'MCD': 0.0,
        'Periodicity': 0.0,
        'V/UV F1': 0.0,
        'dir_results': {},
    }
    for gt_dir, synth_dir in zip(list_gt_dir, list_synth_dir):
        results = evaluate(gt_dir, synth_dir)
        results_tot['M-STFT'] += results['M-STFT']
        results_tot['PESQ'] += results['PESQ']
        results_tot['MCD'] += results['MCD']
        results_tot['Periodicity'] += results['Periodicity']
        results_tot['V/UV F1'] += results['V/UV F1']
        results_tot['dir_results'][synth_dir] = results
    results_tot['M-STFT'] /= len(results_tot['dir_results'])
    results_tot['PESQ'] /= len(results_tot['dir_results'])
    results_tot['MCD'] /= len(results_tot['dir_results'])
    results_tot['Periodicity'] /= len(results_tot['dir_results'])
    results_tot['V/UV F1'] /= len(results_tot['dir_results'])

    # Print to stdout
    print(results_tot)

    if a.output_file:
        # Write results
        with open(a.output_file, 'w') as file:
            json.dump(results_tot, file, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()
