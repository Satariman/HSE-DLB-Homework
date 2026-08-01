import warnings
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate

from src.datasets.data_utils import get_dataloaders
from src.trainer import Inferencer
from src.utils.init_utils import set_random_seed
from src.utils.io_utils import ROOT_PATH

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    """
    Main script for inference. Instantiates the model, metrics, and
    dataloaders. Runs Inferencer to calculate metrics and (or)
    save predictions.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.inferencer.seed)

    if config.inferencer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.inferencer.device

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)

    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    print(model)

    # get metrics
    metrics = instantiate(config.metrics)

    # Prediction saving is optional. Set inferencer.save_path to null when
    # only aggregate metrics (for example, EER) are needed.
    save_path = None
    if config.inferencer.save_path is not None:
        save_path = ROOT_PATH / "data" / "saved" / config.inferencer.save_path
        save_path.mkdir(exist_ok=True, parents=True)

    inferencer = Inferencer(
        model=model,
        config=config,
        device=device,
        dataloaders=dataloaders,
        batch_transforms=batch_transforms,
        save_path=save_path,
        metrics=metrics,
        skip_model_load=False,
    )

    logs = inferencer.run_inference()

    result_lines = []
    for part in logs.keys():
        for key, value in logs[part].items():
            full_key = part + "_" + key
            print(f"    {full_key:15s}: {value}")
            result_lines.append(f"{full_key}: {value}")

    results_path = config.inferencer.get("results_path")
    if results_path is not None:
        results_path = Path(results_path)
        if not results_path.is_absolute():
            results_path = ROOT_PATH / results_path
        results_path.parent.mkdir(exist_ok=True, parents=True)
        results_path.write_text("\n".join(result_lines) + "\n", encoding="utf-8")
        print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
