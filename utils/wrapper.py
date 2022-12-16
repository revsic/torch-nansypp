from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.functional as AF

from config import Config
from disc import Discriminator
from nansypp import Nansypp

from .augment import Augment


class TrainingWrapper:
    """Training wrapper.
    """
    def __init__(self,
                 model: Nansypp,
                 disc: Discriminator,
                 config: Config,
                 device: torch.device):
        """Initializer.
        Args:
            model: NANSY++ model.
            disc: discriminator.
            config: training configurations.
            device: torch device.
        """
        self.model = model
        self.disc = disc
        self.config = config
        self.device = device
        # augmentation
        self.aug = Augment(config)
        self.aug.to(device)
        # alias
        self.seglen = self.config.train.seglen
        self.content_weight = self.config.train.content_start

    def random_segment(self, bunch: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Segment the spectrogram and audio into fixed sized array.
        Args:
            bunch: input tensors.
                sid: [np.long; [B]], speaker id.
                speeches: [np.float32; [B, T]], speeches.
                lengths: [np.long; [B]], speech lengths.
        Returns:
            randomly segmented spectrogram and audios.
        """
        # [B]
        sid, speeches, lengths = bunch
        def segment(seq: np.ndarray, len_: np.ndarray) -> np.ndarray:
            # [B]
            start = np.random.randint(np.maximum(1, len_ - self.seglen))
            # [B, seglen]
            return np.array(
                [np.pad(q[s:s + self.seglen], [0, max(self.seglen - len(q), 0)])
                 for q, s in zip(seq, start)])
        # [B], [B, seglen]
        return sid, segment(speeches, lengths)

    def sample_like(self, signal: torch.Tensor) -> List[torch.Tensor]:
        """Sample augmentation parameters.
        Args:
            signal: [torch.float32; [B, T]], speech signal.
        Returns:
            augmentation parameters.
        """
        # [B]
        bsize, _ = signal.shape
        def sampler(ratio):
            shifts = torch.rand(bsize, device=signal.device) * (ratio - 1.) + 1.
            # flip
            flip = torch.rand(bsize) < 0.5
            shifts[flip] = shifts[flip] ** -1
            return shifts
        # sample shifts
        fs = sampler(self.config.train.formant_shift)
        ps = sampler(self.config.train.pitch_shift)
        # parametric equalizer
        peaks = self.config.train.num_peak
        # quality factor
        power = torch.rand(bsize, peaks + 2, device=signal.device)
        # gains
        g_min, g_max = self.config.train.g_min, self.config.train.g_max
        gain = torch.rand(bsize, peaks, device=signal.device) * (g_max - g_min) + g_min
        return fs, ps, power, gain

    @torch.no_grad()
    def augment(self, signal: torch.Tensor) -> torch.Tensor:
        """Augment the speech.
        Args:
            signal: [torch.float32; [B, T]], segmented speech.
        Returns:
            [torch.float32; [B, T]], speech signal.
        """
        # B
        bsize, _ = signal.shape
        saves = None
        while saves is None or len(saves) < bsize:
            # [B] x 4
            fshift, pshift, power, gain = self.sample_like(signal)
            # [B, T]
            out = self.aug.forward(signal, pshift, fshift, power, gain)
            # for covering unexpected NaN
            nan = out.isnan().any(dim=-1)
            if not nan.all():
                # save the outputs for not-nan inputs
                if saves is None:
                    saves = out[~nan]
                else:
                    saves = torch.cat([saves, out[~nan]], dim=0)
        # [B, T]
        return saves[:bsize]

    def loss_discriminator(self, seg: torch.Tensor) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the discriminator loss.
        Args:
            seg: [torch.float32; [B, T]], segmented speeches.
        Returns:
            loss and disctionaries.
        """
        with torch.no_grad():
            # augmentation
            aug = self.augment(seg)
            # [B, N]
            _, pitch, p_amp, ap_amp = self.model.analyze_pitch(seg)
            # [B, lin_hiddens, S]
            ling = self.model.analyze_linguistic(aug)
            # [B, timb_global], [B, timb_timber, timb_tokens]
            timber_global, timber_bank = self.model.analyze_timber(seg)
            # [B, T], [B, T]
            excit, synth = self.model.synthesize(
                pitch, p_amp, ap_amp, ling, timber_global, timber_bank)
            # truncating
            _, timesteps = seg.shape
            synth = synth[:, :timesteps]

        logits_f, _ = self.disc.forward(synth)
        logits_r, _ = self.disc.forward(seg)

        # discriminative
        d_fake, d_real = 0., 0.
        for logit_f, logit_r in zip(logits_f, logits_r):
            d_fake = d_fake + logit_f.square().mean()
            d_real = d_real + (1 - logit_r).square().mean()

        loss = d_fake + d_real
        losses = {
            'disc/loss': loss.item(),
            'disc/d-real': d_real.mean().item(),
            'disc/d-fake': d_fake.mean().item()}
        return loss, losses, {
            'excit': excit.cpu().detach().numpy(),
            'synth': synth.cpu().detach().numpy()}

    def loss_generator(self, sid: np.ndarray, seg: torch.Tensor) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the generator loss.
        Args:
            sid: [np.long; [B]], speaker id.
            seg: [torch.float32; [B, T]], segmented speech.
        Returns:
            loss and disctionaries.
        """
        # B, T
        bsize, timesteps = seg.shape
        # augmnentation
        aug = self.augment(seg)
        # [B, cqt_bins, N], [B, N]
        cqt, pitch, p_amp, ap_amp = self.model.analyze_pitch(seg)
        # [B, lin_hiddens, S]
        ling = self.model.analyze_linguistic(aug)
        # [B, timb_global], [B, timb_timber, timb_tokens]
        timber_global, timber_bank = self.model.analyze_timber(seg)
        # [B, T], [B, T]
        excit, synth = self.model.synthesize(
            pitch, p_amp, ap_amp, ling, timber_global, timber_bank)
        # truncating
        synth = synth[:, :timesteps]

        # reconstruction loss
        mel_f = self.model.melspec.forward(synth)
        mel_r = self.model.melspec.forward(seg)
        mel_loss = (mel_f - mel_r).abs().mean()

        mss_loss = 0.
        # [B x 2, T]
        bunch = torch.cat([seg, synth], dim=0)
        for hop, win in zip(self.config.train.hops, self.config.train.wins):
            # [B x 2, win // 2 + 1, T / strides]
            fft = torch.stft(
                bunch, win, hop, window=torch.hann_window(win, device=self.device),
                return_complex=True)
            # [B, win // 2 + 1, T / strides]
            mag_f, mag_r = fft.abs().chunk(2, dim=0)
            # []
            mss_loss = mss_loss + (mag_f - mag_r).square().mean()
        # aggregate
        rctor_loss = mel_loss + mss_loss

        # pitch self-supervision
        dist = torch.randint(
            self.config.train.cqt_shift_min,
            self.config.train.cqt_shift_max + 1,  # for inclusive range
            (bsize,), device=self.device)
        # real start index
        start = dist + self.model.cqt_center
        # sampled
        biased = torch.stack([
            cqt_[i:i + self.config.model.pitch_freq]
            for cqt_, i in zip(cqt, start)])
        # [B, N, f0_bins]
        biased_bins, _, _ = self.model.pitch.forward(biased)
        # [B, N]
        biased_pitch = (biased_bins * self.model.pitch_bins).sum(dim=-1)

        # pitch consistency
        pitch_loss = F.huber_loss(
            biased_pitch.log2() + 0.5 * dist[:, None],
            pitch.log2(),
            delta=self.config.train.delta)
        
        # metric purpose
        # [B, N']
        gt_pitch = AF.detect_pitch_frequency(seg, sample_rate=self.config.model.sr)
        # [B, N]
        gt_pitch = F.interpolate(gt_pitch[:, None], size=pitch.shape[-1], mode='linear').squeeze(dim=1)
        # []
        metric_pitch_acc = F.l1_loss(
            pitch.clamp_min(1e-5).log2(),
            gt_pitch.clamp_min(1e-5).log2()).item()


        # metric purpose
        confusion = torch.matmul(timber_global, timber_global.T)
        pos_mask = torch.tensor(
            (sid[:, None] == sid).astype(np.float32) * (1 - np.eye(bsize)),
            device=self.device)
        # placeholder
        metric_timber_pos, metric_timber_neg = None, None
        # positive case
        if pos_mask.sum() > 0:
            metric_timber_pos = (confusion * pos_mask).mean().item()
        # negative case
        if (1 - pos_mask).sum() > 0:
            metric_timber_neg = (confusion * (1 - pos_mask)).mean().item()

        # linguistic informations
        aug2 = self.augment(seg)
        # [B, lin_hiddens, S]
        ling2 = self.model.analyze_linguistic(aug2)
        
        # alias
        kappa = self.config.train.kappa
        n_adj, n_cand = self.config.train.content_adj, self.config.train.candidates
        # [B, N, N]
        confusion = torch.matmul(
            F.normalize(ling, p=2, dim=1).transpose(1, 2),
            F.normalize(ling2, p=2, dim=1))
        # temperature
        confusion = confusion / kappa
        # N
        num_tokens = ling.shape[-1]
        # [N]
        arange = torch.arange(num_tokens)
        # [B, N], positive
        pos = confusion[:, arange, arange]
        # [N]
        placeholder = torch.zeros(num_tokens, device=self.device)
        # [N, N]
        mask = torch.stack([
            placeholder.scatter(
                0,
                (
                    torch.randperm(num_tokens - n_adj, device=self.device)[:n_cand]
                    + i + n_adj // 2 + 1) % num_tokens,
                1.)
            for i in range(num_tokens)])
        # [B, N, N(sum = candidates)], negative case
        masked = confusion.masked_fill(~mask.to(torch.bool), -np.inf)
        # [B, N], negative case
        neg = torch.logsumexp(masked, dim=-1)
        # [B]
        cont_loss = -torch.logsumexp(pos - neg, dim=-1).mean()

        # metric purpose
        metric_pos = pos.mean().item() * kappa
        metric_neg = ((confusion * mask).sum(dim=-1) / n_cand).mean().item() * kappa

        # discriminative
        logits_f, fmaps_f = self.disc.forward(synth)
        _, fmaps_r = self.disc.forward(seg)

        d_fake = 0.
        for logit_f in logits_f:
            d_fake = d_fake + (1 - logit_f).square().mean()

        # feature matching
        fmap_loss = 0.
        for fmap_f, fmap_r in zip(fmaps_f, fmaps_r):
            for ff, fr in zip(fmap_f, fmap_r):
                fmap_loss = fmap_loss + (ff - fr).abs().mean()
        # reweighting
        weight = (rctor_loss / fmap_loss).detach() * fmap_loss

        loss = d_fake + weight * fmap_loss + rctor_loss + pitch_loss + self.content_weight * cont_loss
        losses = {
            'gen/loss': loss.item(),
            'gen/d-fake': d_fake.item(),
            'gen/fmap': fmap_loss.item(),
            'gen/rctor': rctor_loss.item(),
            'gen/pitch': pitch_loss.item(),
            'gen/cont': cont_loss.item(),
            'metric/cont-pos': metric_pos,
            'metric/cont-neg': metric_neg,
            'metric/AFpitch-l1': metric_pitch_acc,
            'common/warmup': self.content_weight,
            'common/weight': weight.item()}
        # conditional ploting
        if metric_timber_pos is not None:
            losses['metric/timber-pos'] = metric_timber_pos
        if metric_timber_neg is not None:
            losses['metric/timber-pos'] = metric_timber_neg

        return loss, losses, {
            'excit': excit.cpu().detach().numpy(),
            'synth': synth.cpu().detach().numpy(),
            'mel_f': mel_f.cpu().detach().numpy(),
            'mel_r': mel_r.cpu().detach().numpy(),
            'log-cqt': cqt.clamp_min(1e-5).log().cpu().detach().numpy(),
            'AFpitch': gt_pitch.clamp_min(1e-5).log2().cpu().detach().numpy(),
            'pitch': pitch.clamp_min(1e-5).log2().cpu().detach().numpy()}

    def update_warmup(self):
        """Update the content loss weights.
        """
        self.content_weight = min(
            self.content_weight + self.config.train.content_start,
            self.config.train.content_end)
