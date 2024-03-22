# Vocoder Evaluation

This repository contains the evaluation tool used in **"BigVSAN: Enhancing GAN-based Neural Vocoders with Slicing Adversarial Network"** (*[arXiv 2309.02836](https://arxiv.org/abs/2309.02836)*).
Please cite [[1](#citation)] in your work when using this code in your experiments.

## Quick Start

First, prepare an environment
```shell
pip install -r requirements.txt
```

Then, perform an evaluation
```shell
python evaluate.py <gt_dir 1> <synth_dir 1> <gt_dir 2> <synth_dir 2> ... <gt_dir N> <synth_dir N>
```
```gt_dir n``` means a directory that contains ground-truth audio files, and ```synth_dir n``` means a directory that contains synthesized audio files. Each file in ```synth_dir n``` needs to have the corresponding file that has the same name in ```gt_dir n```. Also, a corresponding pair needs to be time-aligned in advance.

```evaluate.py``` will output calculated metrics for each ```gt_dir n```-```synth_dir n``` pair and the macro averages of them across all pairs. It will take some time to complete an evaluation.

## Supported evaluation metrics
This toolbox supports the following metrics:

- M-STFT: Multi-resolution short-term Fourier transform
- PESQ: Perceptual evaluation of speech quality
- MCD: Mel-cepstral distortion
- Periodicity: Periodicity error
- V/UV F1: F1 score of voiced/unvoiced classification

## Citation

If you find this tool useful, please consider citing

[1] Shibuya, T., Takida, Y., Mitsufuji, Y.,
"BigVSAN: Enhancing GAN-based Neural Vocoders with Slicing Adversarial Network,"
ICASSP 2024.
```bibtex
@inproceedings{shibuya2024bigvsan,
        title={{BigVSAN}: Enhancing GAN-based Neural Vocoders with Slicing Adversarial Network},
        author={Shibuya, Takashi and Takida, Yuhta and Mitsufuji, Yuki},
        booktitle={ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
        year={2024}
}
```

## References

> https://github.com/NVIDIA/BigVGAN

> https://github.com/csteinmetz1/auraloss

> https://github.com/ludlows/PESQ

> https://github.com/ttslr/python-MCD

> https://github.com/descriptinc/cargan
