import torch

class TrimPad(torch.nn.Module):
    def __init__(self, target_frames=750):
        super().__init__()
        self.target_frames = target_frames

    def forward(self, x):
        n_frames = x.shape[-2]

        if n_frames > self.target_frames:
            max_start = n_frames - self.target_frames
            start = torch.randint(0, max_start + 1, ()).item()

            return x[..., start:start + self.target_frames, :]

        if n_frames < self.target_frames:
            pad = self.target_frames - n_frames

            return torch.nn.functional.pad(x,(0, 0, 0, pad), value=0.0,)

        return x