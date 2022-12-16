import argparse
import json
import os

import git
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter

import speechset
from config import Config
from disc import Discriminator
from nansypp import Nansypp
from speechset.utils.melstft import MelSTFT
from utils.dump import DumpReader
from utils.wrapper import TrainingWrapper


class Trainer:
    """NANSY++ trainer.
    """
    LOG_IDX = 0
    LOG_MAXLEN = 1.5
    LOG_AUDIO = 3

    def __init__(self,
                 model: Nansypp,
                 disc: Discriminator,
                 dataset: speechset.WavDataset,
                 testset: speechset.WavDataset,
                 config: Config,
                 device: torch.device):
        """Initializer.
        Args:
            model: NANSY++ model.
            disc: discriminator.
            dataset, testset: dataset.
            config: unified configurations.
            device: target computing device.
        """
        self.model = model
        self.disc = disc
        self.config = config

        self.dataset = dataset
        self.testset = testset

        self.loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=config.train.batch,
            shuffle=config.train.shuffle,
            collate_fn=self.dataset.collate,
            num_workers=config.train.num_workers,
            pin_memory=config.train.pin_memory)

        self.testloader = torch.utils.data.DataLoader(
            self.testset,
            batch_size=config.train.batch,
            collate_fn=self.dataset.collate,
            num_workers=config.train.num_workers,
            pin_memory=config.train.pin_memory)

        # training wrapper
        self.wrapper = TrainingWrapper(model, disc, config, device)

        self.optim_g = torch.optim.Adam(
            self.model.parameters(),
            config.train.learning_rate_g,
            (config.train.beta1, config.train.beta2))

        self.optim_d = torch.optim.Adam(
            self.disc.parameters(),
            config.train.learning_rate_d,
            (config.train.beta1, config.train.beta2))

        self.train_log = SummaryWriter(
            os.path.join(config.train.log, config.train.name, 'train'))
        self.test_log = SummaryWriter(
            os.path.join(config.train.log, config.train.name, 'test'))

        self.ckpt_path = os.path.join(
            config.train.ckpt, config.train.name, config.train.name)

        self.melspec = MelSTFT(config.data)
        self.cmap = np.array(plt.get_cmap('viridis').colors)

    def train(self, epoch: int = 0):
        """Train wavegrad.
        Args:
            epoch: starting step.
        """
        self.model.train()
        step = epoch * len(self.loader)
        for epoch in tqdm.trange(epoch, self.config.train.epoch):
            with tqdm.tqdm(total=len(self.loader), leave=False) as pbar:
                for it, bunch in enumerate(self.loader):
                    sid, seg = self.wrapper.random_segment(bunch)
                    seg = torch.tensor(seg, device=self.wrapper.device)
                    loss_g, losses_g, aux_g = self.wrapper.loss_generator(sid, seg)
                    # update
                    self.optim_g.zero_grad()
                    loss_g.backward()
                    self.optim_g.step()

                    loss_d, losses_d, _ = self.wrapper.loss_discriminator(seg)
                    # update
                    self.optim_d.zero_grad()
                    loss_d.backward()
                    self.optim_d.step()

                    step += 1
                    pbar.update()
                    pbar.set_postfix({'loss': loss_d.item(), 'step': step})

                    self.wrapper.update_warmup()

                    for key, val in {**losses_g, **losses_d}.items():
                        self.train_log.add_scalar(key, val, step)

                    with torch.no_grad():
                        grad_norm = np.mean([
                            torch.norm(p.grad).item()
                            for p in self.model.parameters() if p.grad is not None])
                        param_norm = np.mean([
                            torch.norm(p).item()
                            for p in self.model.parameters() if p.dtype == torch.float32])

                    self.train_log.add_scalar('common/grad-norm', grad_norm, step)
                    self.train_log.add_scalar('common/param-norm', param_norm, step)
                    self.train_log.add_scalar(
                        'common/learning-rate-g', self.optim_g.param_groups[0]['lr'], step)
                    self.train_log.add_scalar(
                        'common/learning-rate-d', self.optim_d.param_groups[0]['lr'], step)

                    if it % (len(self.loader) // 50) == 0:
                        self.train_log.add_image(
                            # [3, M, T]
                            'mel-gt/train', self.mel_img(aux_g['mel_r'][Trainer.LOG_IDX]), step)
                        self.train_log.add_image(
                            # [3, M, T]
                            'mel-synth/train', self.mel_img(aux_g['mel_f'][Trainer.LOG_IDX]), step)
                        self.train_log.add_image(
                            # [3, M, T]
                            'log-cqt/train', self.mel_img(aux_g['log-cqt'][Trainer.LOG_IDX]), step)
                        self.train_log.add_audio(
                            'speech/train', seg.cpu().numpy()[Trainer.LOG_IDX, None], step,
                            sample_rate=self.config.data.sr)
                        self.train_log.add_audio(
                            'synth/train', aux_g['synth'][Trainer.LOG_IDX, None], step,
                            sample_rate=self.config.data.sr)
                        # pitch plot
                        fig = plt.figure()
                        ax = fig.add_subplot(1, 1, 1)
                        ax.plot(aux_g['AFpitch'][Trainer.LOG_IDX], label='log-AFpitch')
                        ax.plot(aux_g['pitch'][Trainer.LOG_IDX], label='log-pitch')
                        ax.legend()
                        self.train_log.add_figure('pitch/train', fig, step)

            losses = {
                key: [] for key in {**losses_d, **losses_g}}
            with torch.no_grad():
                for bunch in tqdm.tqdm(self.testloader, leave=False):
                    sid, seg = self.wrapper.random_segment(bunch)
                    seg = torch.tensor(seg, device=self.wrapper.device)
                    _, losses_g, _ = self.wrapper.loss_generator(sid, seg)
                    _, losses_d, _ = self.wrapper.loss_discriminator(seg)
                    for key, val in {**losses_g, **losses_d}.items():
                        losses[key].append(val)

                # test log
                for key, val in losses.items():
                    self.test_log.add_scalar(key, np.mean(val), step)

                # wrap last bunch
                _, speeches, lengths = bunch
                # B
                bsize, = lengths.shape
                # inference
                self.model.eval()
                for i in range(Trainer.LOG_AUDIO):
                    idx = (Trainer.LOG_IDX + i) % bsize
                    # min-length
                    len_ = min(
                        lengths[idx].item(),
                        int(Trainer.LOG_MAXLEN * self.config.model.sr))
                    # [T], gt plot
                    speech = speeches[idx, :len_]
                    self.test_log.add_image(
                        f'mel-gt/test{i}', self.mel_img(self.melspec(speech).T), step)
                    self.test_log.add_audio(
                        f'speech/test{i}', speech[None], step, sample_rate=self.config.data.sr)

                    # [1, T]
                    synth, _ = self.model.forward(
                        torch.tensor(speech[None], device=self.wrapper.device))

                    synth = synth.squeeze(dim=0).cpu().numpy()
                    self.test_log.add_image(
                        f'mel-synth/test{i}', self.mel_img(self.melspec(synth).T), step)
                    self.test_log.add_audio(
                        f'synth/test{i}', synth[None], step, sample_rate=self.config.data.sr)

                self.model.train()

            self.model.save(f'{self.ckpt_path}_{epoch}.ckpt', self.optim_g)
            self.disc.save(f'{self.ckpt_path}_{epoch}.ckpt-disc', self.optim_d)

    def mel_img(self, mel: np.ndarray) -> np.ndarray:
        """Generate mel-spectrogram images.
        Args:
            mel: [np.float32; [mel, T]], mel-spectrogram.
        Returns:
            [np.float32; [3, mel, T]], mel-spectrogram in viridis color map.
        """
        # minmax norm in range(0, 1)
        mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-7)
        # in range(0, 255)
        mel = (mel * 255).astype(np.uint8)
        # [mel, T, 3]
        mel = self.cmap[mel]
        # [3, mel, T], make origin lower
        mel = np.flip(mel, axis=0).transpose(2, 0, 1)
        return mel


if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--config', default=None)
    parser.add_argument('--load-epoch', default=None, type=int)
    parser.add_argument('--name', default=None)
    parser.add_argument('--auto-rename', default=False, action='store_true')
    args = parser.parse_args()

    # seed setting
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # configurations
    config = Config()
    if args.config is not None:
        print('[*] load config: ' + args.config)
        with open(args.config) as f:
            config = Config.load(json.load(f))

    if args.name is not None:
        config.train.name = args.name

    log_path = os.path.join(config.train.log, config.train.name)
    # auto renaming
    if args.auto_rename and os.path.exists(log_path):
        config.train.name = next(
            f'{config.train.name}_{i}' for i in range(1024)
            if not os.path.exists(f'{log_path}_{i}'))
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    ckpt_path = os.path.join(config.train.ckpt, config.train.name)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    sr = config.model.sr
    # prepare datasets
    # trainset = speechset.WavDataset(
    #     speechset.datasets.ConcatReader([
    #         speechset.datasets.LibriTTS('./datasets/LibriTTS/train-clean-100', sr),
    #         speechset.datasets.LibriTTS('./datasets/LibriTTS/train-clean-360', sr),
    #         speechset.datasets.LibriSpeech('./datasets/LibriSpeech/train-other-500', sr),
    #         speechset.datasets.VCTK('./datasets/VCTK-Corpus', sr)]))

    trainset = speechset.utils.IDWrapper(
        speechset.WavDataset(DumpReader('./datasets/dumped')))
    testset = speechset.utils.IDWrapper(
        speechset.WavDataset(DumpReader('./datasets/libri_test_clean')))

    # model definition
    device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Nansypp(config.model)
    model.to(device)

    disc = Discriminator(config.disc)
    disc.to(device)

    trainer = Trainer(model, disc, trainset, testset, config, device)

    # loading
    if args.load_epoch is not None:
        # find checkpoint
        ckpt_path = os.path.join(
            config.train.ckpt,
            config.train.name,
            f'{config.train.name}_{args.load_epoch}.ckpt')
        # load checkpoint
        ckpt = torch.load(ckpt_path)
        model.load_(ckpt, trainer.optim_g)
        # discriminator checkpoint
        ckpt_disc = torch.load(f'{ckpt_path}-disc')
        disc.load_(ckpt_disc, trainer.optim_d)
        print('[*] load checkpoint: ' + ckpt_path)
        # since epoch starts with 0
        args.load_epoch += 1

    # git configuration
    repo = git.Repo()
    config.train.hash = repo.head.object.hexsha
    with open(os.path.join(config.train.ckpt, config.train.name + '.json'), 'w') as f:
        json.dump(config.dump(), f)

    # start train
    trainer.train(args.load_epoch or 0)
