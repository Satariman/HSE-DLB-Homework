import torch
import torchaudio
from torch import nn


class LFCCEnergy(nn.Module):


    def __init__(
        self,
        smpl_rate,
        frm_length_ms,
        frm_shift_ms,
        n_fft,
        n_filter,
        n_lfcc,
        log_lf,
        eps=1e-10,
    ):
        super().__init__()

        win_length = round(smpl_rate * frm_length_ms / 1000)
        hop_length = round(smpl_rate * frm_shift_ms / 1000)

        spectrogram_kwargs = {
            "n_fft": n_fft,
            "win_length": win_length,
            "hop_length": hop_length,
        }

        self.lfcc = torchaudio.transforms.LFCC(
            sample_rate=smpl_rate,
            n_filter=n_filter,
            n_lfcc=n_lfcc,
            log_lf=log_lf,
            speckwargs=spectrogram_kwargs,
        )

        self.spectrogram = torchaudio.transforms.Spectrogram(
            **spectrogram_kwargs
        )

        self.eps = eps

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        lfcc = self.lfcc(waveform)
        spec = self.spectrogram(waveform)
        lg_spec_energy = torch.log(
            spec.sum(dim=-2) + self.eps
        )

        lfcc = lfcc.clone()
        lfcc[..., 0, :] = lg_spec_energy

        return lfcc