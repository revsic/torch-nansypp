# torch-nansypp

Torch implementation of NANSY++: Unified Voice Synthesis with Neural Analysis and Synthesis, [[openreview](https://openreview.net/forum?id=elDEe8LYW7-)]

## TODO

1. breathiness perturbation
2. DEMAND based noise addition

## Requirements

Tested in python 3.7.9 conda environment.

## Usage

Initialize the submodule.

```bash
git submodule init --update
```

Download LibriTTS[[openslr:60](https://www.openslr.org/60/)], LibriSpeech[[openslr:12](https://www.openslr.org/12)] and VCTK[[official](https://datashare.ed.ac.uk/handle/10283/2651)] datasets.

Dump the dataset for training.

```
python -m speechset.utils.dump \
    --out-dir ./datasets/dumped
```

To train model, run [train.py](./train.py)

```bash
python train.py
```

To start to train from previous checkpoint, `--load-epoch` is available.

```bash
python train.py \
    --load-epoch 20 \
    --config ./ckpt/t1.json
```

Checkpoint will be written on TrainConfig.ckpt, tensorboard summary on TrainConfig.log.

```bash
tensorboard --logdir ./log
```

[TODO] To inference model, run [inference.py](./inference.py)


[TODO] Pretrained checkpoints will be relased on [releases](https://github.com/revsic/torch-nansypp/releases).

To use pretrained model, download files and unzip it. Followings are sample script.

```py
from nansypp import Nansypp

ckpt = torch.load('t1_200.ckpt', map_location='cpu')
nansypp = Nansypp.load(ckpt)
nansy.eval()
```

## [TODO] Learning curve and Figures

## [TODO] Samples
