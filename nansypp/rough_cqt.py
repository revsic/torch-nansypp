"""
origin: https://github.com/KinWaiCheuk/nnAudio
commit: dc222b438ecac32492354a2df0b4faa24f473cc2
(last commited in Oct 9, 2022)

---
MIT License

Copyright (c) 2019 KinWaiCheuk

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import warnings

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal


def nextpow2(A):
    """A helper function to calculate the next nearest number to the power of 2.
    Parameters
    ----------
    A : float
        A float number that is going to be rounded up to the nearest power of 2
    Returns
    -------
    int
        The nearest power of 2 to the input number ``A``
    Examples
    --------
    >>> nextpow2(6)
    3
    """

    return int(np.ceil(np.log2(A)))


def create_lowpass_filter(band_center=0.5, kernelLength=256, transitionBandwidth=0.03):
    """
    Calculate the highest frequency we need to preserve and the lowest frequency we allow
    to pass through.
    Note that frequency is on a scale from 0 to 1 where 0 is 0 and 1 is Nyquist frequency of
    the signal BEFORE downsampling.
    """

    # transitionBandwidth = 0.03
    passbandMax = band_center / (1 + transitionBandwidth)
    stopbandMin = band_center * (1 + transitionBandwidth)

    # Unlike the filter tool we used online yesterday, this tool does
    # not allow us to specify how closely the filter matches our
    # specifications. Instead, we specify the length of the kernel.
    # The longer the kernel is, the more precisely it will match.
    # kernelLength = 256

    # We specify a list of key frequencies for which we will require
    # that the filter match a specific output gain.
    # From [0.0 to passbandMax] is the frequency range we want to keep
    # untouched and [stopbandMin, 1.0] is the range we want to remove
    keyFrequencies = [0.0, passbandMax, stopbandMin, 1.0]

    # We specify a list of output gains to correspond to the key
    # frequencies listed above.
    # The first two gains are 1.0 because they correspond to the first
    # two key frequencies. the second two are 0.0 because they
    # correspond to the stopband frequencies
    gainAtKeyFrequencies = [1.0, 1.0, 0.0, 0.0]

    # This command produces the filter kernel coefficients
    filterKernel = signal.firwin2(kernelLength, keyFrequencies, gainAtKeyFrequencies)

    return filterKernel.astype(np.float32)


def create_cqt_kernels(
    Q,
    fs,
    fmin,
    n_bins=84,
    bins_per_octave=12,
    norm=1,
    window="hann",
    fmax=None,
    topbin_check=True,
    gamma=0,
    pad_fft=True
):
    """
    Automatically create CQT kernels in time domain
    """

    fftLen = 2 ** nextpow2(np.ceil(Q * fs / fmin))
    # minWin = 2**nextpow2(np.ceil(Q * fs / fmax))

    if (fmax != None) and (n_bins == None):
        n_bins = np.ceil(
            bins_per_octave * np.log2(fmax / fmin)
        )  # Calculate the number of bins
        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.float(bins_per_octave))

    elif (fmax == None) and (n_bins != None):
        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.float(bins_per_octave))

    else:
        warnings.warn("If fmax is given, n_bins will be ignored", SyntaxWarning)
        n_bins = np.ceil(
            bins_per_octave * np.log2(fmax / fmin)
        )  # Calculate the number of bins
        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.float(bins_per_octave))

    if np.max(freqs) > fs / 2 and topbin_check == True:
        raise ValueError(
            "The top bin {}Hz has exceeded the Nyquist frequency, \
                          please reduce the n_bins".format(
                np.max(freqs)
            )
        )

    alpha = 2.0 ** (1.0 / bins_per_octave) - 1.0
    lengths = np.ceil(Q * fs / (freqs + gamma / alpha))
    
    # get max window length depending on gamma value
    max_len = int(max(lengths))
    fftLen = int(2 ** (np.ceil(np.log2(max_len))))

    tempKernel = np.zeros((int(n_bins), int(fftLen)), dtype=np.complex64)
    specKernel = np.zeros((int(n_bins), int(fftLen)), dtype=np.complex64)

    for k in range(0, int(n_bins)):
        freq = freqs[k]
        l = lengths[k]

        # Centering the kernels
        if l % 2 == 1:  # pad more zeros on RHS
            start = int(np.ceil(fftLen / 2.0 - l / 2.0)) - 1
        else:
            start = int(np.ceil(fftLen / 2.0 - l / 2.0))

        window_dispatch = librosa.filters.get_window(window, int(l), fftbins=True)
        sig = window_dispatch * np.exp(np.r_[-l // 2 : l // 2] * 1j * 2 * np.pi * freq / fs) / l

        if norm:  # Normalizing the filter # Trying to normalize like librosa
            tempKernel[k, start : start + int(l)] = sig / np.linalg.norm(sig, norm)
        else:
            tempKernel[k, start : start + int(l)] = sig
        # specKernel[k, :] = fft(tempKernel[k])

    # return specKernel[:,:fftLen//2+1], fftLen, torch.tensor(lenghts).float()
    return tempKernel, fftLen, torch.tensor(lengths).float(), freqs


# The following two downsampling count functions are obtained from librosa CQT
# They are used to determine the number of pre resamplings if the starting and ending frequency
# are both in low frequency regions.
def early_downsample_count(nyquist, filter_cutoff, hop_length, n_octaves):
    """Compute the number of early downsampling operations"""

    downsample_count1 = max(
        0, int(np.ceil(np.log2(0.85 * nyquist / filter_cutoff)) - 1) - 1
    )
    # print("downsample_count1 = ", downsample_count1)
    num_twos = nextpow2(hop_length)
    downsample_count2 = max(0, num_twos - n_octaves + 1)
    # print("downsample_count2 = ",downsample_count2)

    return min(downsample_count1, downsample_count2)


def early_downsample(sr, hop_length, n_octaves, nyquist, filter_cutoff):
    """Return new sampling rate and hop length after early dowansampling"""
    downsample_count = early_downsample_count(
        nyquist, filter_cutoff, hop_length, n_octaves
    )
    # print("downsample_count = ", downsample_count)
    downsample_factor = 2 ** (downsample_count)

    hop_length //= downsample_factor  # Getting new hop_length
    new_sr = sr / float(downsample_factor)  # Getting new sampling rate
    sr = new_sr

    return sr, hop_length, downsample_factor


def get_early_downsample_params(sr, hop_length, fmax_t, Q, n_octaves, verbose):
    """Used in CQT2010 and CQT2010v2"""

    window_bandwidth = 1.5  # for hann window
    filter_cutoff = fmax_t * (1 + 0.5 * window_bandwidth / Q)
    sr, hop_length, downsample_factor = early_downsample(
        sr, hop_length, n_octaves, sr // 2, filter_cutoff
    )
    if downsample_factor != 1:
        if verbose == True:
            print("Can do early downsample, factor = ", downsample_factor)
        earlydownsample = True
        # print("new sr = ", sr)
        # print("new hop_length = ", hop_length)
        early_downsample_filter = create_lowpass_filter(
            band_center=1 / downsample_factor,
            kernelLength=256,
            transitionBandwidth=0.03,
        )
        early_downsample_filter = torch.tensor(early_downsample_filter)[None, None, :]

    else:
        if verbose == True:
            print(
                "No early downsampling is required, downsample_factor = ",
                downsample_factor,
            )
        early_downsample_filter = None
        earlydownsample = False

    return sr, hop_length, downsample_factor, early_downsample_filter, earlydownsample


def get_cqt_complex(x, cqt_kernels_real, cqt_kernels_imag, hop_length, padding):
    """Multiplying the STFT result with the cqt_kernel, check out the 1992 CQT paper [1]
    for how to multiple the STFT result with the CQT kernel
    [2] Brown, Judith C.C. and Miller Puckette. “An efficient algorithm for the calculation of
    a constant Q transform.” (1992)."""

    # STFT, converting the audio input from time domain to frequency domain
    try:
        x = padding(
            x
        )  # When center == True, we need padding at the beginning and ending
    except:
        warnings.warn(
            f"\ninput size = {x.shape}\tkernel size = {cqt_kernels_real.shape[-1]}\n"
            "padding with reflection mode might not be the best choice, try using constant padding",
            UserWarning,
        )
        x = torch.nn.functional.pad(
            x, (cqt_kernels_real.shape[-1] // 2, cqt_kernels_real.shape[-1] // 2)
        )
    CQT_real = F.conv1d(x, cqt_kernels_real, stride=hop_length)
    CQT_imag = -F.conv1d(x, cqt_kernels_imag, stride=hop_length)

    return torch.stack((CQT_real, CQT_imag), -1)


def downsampling_by_n(x, filterKernel, n):
    """A helper function that downsamples the audio by a arbitary factor n.
    It is used in CQT2010 and CQT2010v2.
    Parameters
    ----------
    x : torch.Tensor
        The input waveform in ``torch.Tensor`` type with shape ``(batch, 1, len_audio)``
    filterKernel : str
        Filter kernel in ``torch.Tensor`` type with shape ``(1, 1, len_kernel)``
    n : int
        The downsampling factor
    Returns
    -------
    torch.Tensor
        The downsampled waveform
    Examples
    --------
    >>> x_down = downsampling_by_n(x, filterKernel)
    """

    x = F.conv1d(x, filterKernel, stride=n, padding=(filterKernel.shape[-1] - 1) // 2)
    return x


def downsampling_by_2(x, filterKernel):
    """A helper function that downsamples the audio by half. It is used in CQT2010 and CQT2010v2
    Parameters
    ----------
    x : torch.Tensor
        The input waveform in ``torch.Tensor`` type with shape ``(batch, 1, len_audio)``
    filterKernel : str
        Filter kernel in ``torch.Tensor`` type with shape ``(1, 1, len_kernel)``
    Returns
    -------
    torch.Tensor
        The downsampled waveform
    Examples
    --------
    >>> x_down = downsampling_by_2(x, filterKernel)
    """

    x = F.conv1d(x, filterKernel, stride=2, padding=(filterKernel.shape[-1] - 1) // 2)
    return x


class CQT2010v2(nn.Module):
    """This function is to calculate the CQT of the input signal.
    Input signal should be in either of the following shapes.\n
    1. ``(len_audio)``\n
    2. ``(num_audio, len_audio)``\n
    3. ``(num_audio, 1, len_audio)``
    The correct shape will be inferred autommatically if the input follows these 3 shapes.
    Most of the arguments follow the convention from librosa.
    This class inherits from ``nn.Module``, therefore, the usage is same as ``nn.Module``.
    This alogrithm uses the resampling method proposed in [1].
    Instead of convoluting the STFT results with a gigantic CQT kernel covering the full frequency
    spectrum, we make a small CQT kernel covering only the top octave. Then we keep downsampling the
    input audio by a factor of 2 to convoluting it with the small CQT kernel.
    Everytime the input audio is downsampled, the CQT relative to the downsampled input is equivalent
    to the next lower octave.
    The kernel creation process is still same as the 1992 algorithm. Therefore, we can reuse the
    code from the 1992 alogrithm [2]
    [1] Schörkhuber, Christian. “CONSTANT-Q TRANSFORM TOOLBOX FOR MUSIC PROCESSING.” (2010).
    [2] Brown, Judith C.C. and Miller Puckette. “An efficient algorithm for the calculation of a
    constant Q transform.” (1992).
    Early downsampling factor is to downsample the input audio to reduce the CQT kernel size.
    The result with and without early downsampling are more or less the same except in the very low
    frequency region where freq < 40Hz.
    Parameters
    ----------
    sr : int
        The sampling rate for the input audio. It is used to calucate the correct ``fmin`` and ``fmax``.
        Setting the correct sampling rate is very important for calculating the correct frequency.
    hop_length : int
        The hop (or stride) size. Default value is 512.
    fmin : float
        The frequency for the lowest CQT bin. Default is 32.70Hz, which coresponds to the note C0.
    fmax : float
        The frequency for the highest CQT bin. Default is ``None``, therefore the higest CQT bin is
        inferred from the ``n_bins`` and ``bins_per_octave``.  If ``fmax`` is not ``None``, then the
        argument ``n_bins`` will be ignored and ``n_bins`` will be calculated automatically.
        Default is ``None``
    n_bins : int
        The total numbers of CQT bins. Default is 84. Will be ignored if ``fmax`` is not ``None``.
    bins_per_octave : int
        Number of bins per octave. Default is 12.
    norm : bool
        Normalization for the CQT result.
    basis_norm : int
        Normalization for the CQT kernels. ``1`` means L1 normalization, and ``2`` means L2 normalization.
        Default is ``1``, which is same as the normalization used in librosa.
    window : str
        The windowing function for CQT. It uses ``scipy.signal.get_window``, please refer to
        scipy documentation for possible windowing functions. The default value is 'hann'
    pad_mode : str
        The padding method. Default value is 'reflect'.
    trainable : bool
        Determine if the CQT kernels are trainable or not. If ``True``, the gradients for CQT kernels
        will also be caluclated and the CQT kernels will be updated during model training.
        Default value is ``False``
    output_format : str
        Determine the return type.
        'Magnitude' will return the magnitude of the STFT result, shape = ``(num_samples, freq_bins, time_steps)``;
        'Complex' will return the STFT result in complex number, shape = ``(num_samples, freq_bins, time_steps, 2)``;
        'Phase' will return the phase of the STFT reuslt, shape = ``(num_samples, freq_bins,time_steps, 2)``.
        The complex number is stored as ``(real, imag)`` in the last axis. Default value is 'Magnitude'.
    verbose : bool
        If ``True``, it shows layer information. If ``False``, it suppresses all prints.
    Returns
    -------
    spectrogram : torch.tensor
    It returns a tensor of spectrograms.
    shape = ``(num_samples, freq_bins,time_steps)`` if ``output_format='Magnitude'``;
    shape = ``(num_samples, freq_bins,time_steps, 2)`` if ``output_format='Complex' or 'Phase'``;
    Examples
    --------
    >>> spec_layer = Spectrogram.CQT2010v2()
    >>> specs = spec_layer(x)
    """

    # To DO:
    # need to deal with the filter and other tensors

    def __init__(
        self,
        sr=22050,
        hop_length=512,
        fmin=32.70,
        fmax=None,
        n_bins=84,
        filter_scale=1,
        bins_per_octave=12,
        norm=True,
        basis_norm=1,
        window="hann",
        pad_mode="reflect",
        earlydownsample=True,
        trainable=False,
        output_format="Magnitude",
    ):

        super().__init__()

        self.norm = (
            norm  # Now norm is used to normalize the final CQT result by dividing n_fft
        )
        # basis_norm is for normalizing basis
        self.hop_length = hop_length
        self.pad_mode = pad_mode
        self.n_bins = n_bins
        self.earlydownsample = (
            earlydownsample  # We will activate early downsampling later if possible
        )
        self.trainable = trainable
        self.output_format = output_format

        # It will be used to calculate filter_cutoff and creating CQT kernels
        Q = float(filter_scale) / (2 ** (1 / bins_per_octave) - 1)

        # Creating lowpass filter and make it a torch tensor
        lowpass_filter = torch.tensor(
            create_lowpass_filter(
                band_center=0.50, kernelLength=256, transitionBandwidth=0.001
            )
        )

        # Broadcast the tensor to the shape that fits conv1d
        self.register_buffer("lowpass_filter", lowpass_filter[None, None, :])

        # Caluate num of filter requires for the kernel
        # n_octaves determines how many resampling requires for the CQT
        n_filters = min(bins_per_octave, n_bins)
        self.n_octaves = int(np.ceil(float(n_bins) / bins_per_octave))

        # Calculate the lowest frequency bin for the top octave kernel
        self.fmin_t = fmin * 2 ** (self.n_octaves - 1)
        remainder = n_bins % bins_per_octave
        # print("remainder = ", remainder)

        if remainder == 0:
            # Calculate the top bin frequency
            fmax_t = self.fmin_t * 2 ** ((bins_per_octave - 1) / bins_per_octave)
        else:
            # Calculate the top bin frequency
            fmax_t = self.fmin_t * 2 ** ((remainder - 1) / bins_per_octave)

        self.fmin_t = fmax_t / 2 ** (
            1 - 1 / bins_per_octave
        )  # Adjusting the top minium bins
        if fmax_t > sr / 2:
            raise ValueError(
                "The top bin {}Hz has exceeded the Nyquist frequency, \
                            please reduce the n_bins".format(
                    fmax_t
                )
            )

        if (
            self.earlydownsample == True
        ):  # Do early downsampling if this argument is True
            (
                sr,
                self.hop_length,
                self.downsample_factor,
                early_downsample_filter,
                self.earlydownsample,
            ) = get_early_downsample_params(
                sr, hop_length, fmax_t, Q, self.n_octaves, False
            )
            self.register_buffer("early_downsample_filter", early_downsample_filter)
        else:
            self.downsample_factor = 1.0

        # Preparing CQT kernels
        basis, self.n_fft, lenghts, _ = create_cqt_kernels(
            Q,
            sr,
            self.fmin_t,
            n_filters,
            bins_per_octave,
            norm=basis_norm,
            topbin_check=False,
        )
        # For normalization in the end
        # The freqs returned by create_cqt_kernels cannot be used
        # Since that returns only the top octave bins
        # We need the information for all freq bin
        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.float(bins_per_octave))
        self.frequencies = freqs

        lenghts = np.ceil(Q * sr / freqs)
        lenghts = torch.tensor(lenghts).float()
        self.register_buffer("lenghts", lenghts)

        self.basis = basis
        # These cqt_kernel is already in the frequency domain
        cqt_kernels_real = torch.tensor(basis.real).unsqueeze(1)
        cqt_kernels_imag = torch.tensor(basis.imag).unsqueeze(1)

        if trainable:
            cqt_kernels_real = nn.Parameter(cqt_kernels_real, requires_grad=trainable)
            cqt_kernels_imag = nn.Parameter(cqt_kernels_imag, requires_grad=trainable)
            self.register_parameter("cqt_kernels_real", cqt_kernels_real)
            self.register_parameter("cqt_kernels_imag", cqt_kernels_imag)
        else:
            self.register_buffer("cqt_kernels_real", cqt_kernels_real)
            self.register_buffer("cqt_kernels_imag", cqt_kernels_imag)

        # If center==True, the STFT window will be put in the middle, and paddings at the beginning
        # and ending are required.
        if self.pad_mode == "constant":
            self.padding = nn.ConstantPad1d(self.n_fft // 2, 0)
        elif self.pad_mode == "reflect":
            self.padding = nn.ReflectionPad1d(self.n_fft // 2)

    def forward(self, x, output_format=None, normalization_type="librosa"):
        """
        Convert a batch of waveforms to CQT spectrograms.
        Parameters
        ----------
        x : torch tensor
            Input signal should be in either of the following shapes.\n
            1. ``(len_audio)``\n
            2. ``(num_audio, len_audio)``\n
            3. ``(num_audio, 1, len_audio)``
            It will be automatically broadcast to the right shape
        """
        output_format = output_format or self.output_format

        if self.earlydownsample == True:
            x = downsampling_by_n(
                x, self.early_downsample_filter, self.downsample_factor
            )
        hop = self.hop_length
        CQT = get_cqt_complex(
            x, self.cqt_kernels_real, self.cqt_kernels_imag, hop, self.padding
        )  # Getting the top octave CQT

        x_down = x  # Preparing a new variable for downsampling

        for i in range(self.n_octaves - 1):
            hop = hop // 2
            x_down = downsampling_by_2(x_down, self.lowpass_filter)
            CQT1 = get_cqt_complex(
                x_down, self.cqt_kernels_real, self.cqt_kernels_imag, hop, self.padding
            )
            CQT = torch.cat((CQT1, CQT), 1)

        CQT = CQT[:, -self.n_bins :, :]  # Removing unwanted bottom bins
        # print("downsample_factor = ",self.downsample_factor)
        # print(CQT.shape)
        # print(self.lenghts.view(-1,1).shape)

        # Normalizing the output with the downsampling factor, 2**(self.n_octaves-1) is make it
        # same mag as 1992
        CQT = CQT * self.downsample_factor
        # Normalize again to get same result as librosa
        if normalization_type == "librosa":
            CQT = CQT * torch.sqrt(self.lenghts.view(-1, 1, 1))
        elif normalization_type == "convolutional":
            pass
        elif normalization_type == "wrap":
            CQT *= 2
        else:
            raise ValueError(
                "The normalization_type %r is not part of our current options."
                % normalization_type
            )

        if output_format == "Magnitude":
            if self.trainable == False:
                # Getting CQT Amplitude
                return torch.sqrt(CQT.pow(2).sum(-1))
            else:
                return torch.sqrt(CQT.pow(2).sum(-1) + 1e-8)

        elif output_format == "Complex":
            return CQT

        elif output_format == "Phase":
            phase_real = torch.cos(torch.atan2(CQT[:, :, :, 1], CQT[:, :, :, 0]))
            phase_imag = torch.sin(torch.atan2(CQT[:, :, :, 1], CQT[:, :, :, 0]))
            return torch.stack((phase_real, phase_imag), -1)
