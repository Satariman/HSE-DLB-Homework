from pathlib import Path

import torch
import torchaudio

from src.datasets.base_dataset import BaseDataset


class ASVspoof2019LADataset(BaseDataset):

    LABEL_TO_ID = {"bonafide": 0, "spoof": 1}

    def __init__(
        self,
        audio_dir: str | Path,
        protocol_path: str | Path,
        *args,
        **kwargs,
    ):
        self.audio_dir = Path(audio_dir)
        self.protocol_path = Path(protocol_path)

        super().__init__(self._create_index(), *args, **kwargs)

    def _create_index(self) -> list[dict]:
        index = []

        with self.protocol_path.open("r", encoding="utf-8") as protocol_file:
            for line in protocol_file:
                fields = line.split()
                index.append(
                    {
                        "path": str(self.audio_dir / f"{fields[1]}.flac"),
                        "label": self.LABEL_TO_ID[fields[4]],
                    }
                )

        return index

    def load_object(self, path: str) -> torch.Tensor:
        waveform, _ = torchaudio.load(path)
        return waveform
