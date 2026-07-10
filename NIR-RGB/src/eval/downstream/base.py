"""Base evaluator for downstream tasks.

Each evaluator runs a pretrained model on three images per sample (NIR, the
translator's RGB output, ground-truth RGB) and reports:
  - per-condition stats (e.g. detection rate, mean confidence)
  - vs-reference metrics, treating the GT-RGB output as pseudo-ground-truth

The headline number is `gap_closure`: the fraction of the gap between
"pretrained model on NIR direct" and "pretrained model on real RGB" that
the translator closes. 1.0 = matches RGB. 0 = no improvement.
"""

from abc import ABC, abstractmethod


class BaseEvaluator(ABC):
    name: str = ""
    requires: tuple = ()
    primary_metric: str = ""

    @abstractmethod
    def setup(self, device: str):
        """Lazy-load the pretrained model. Called once."""

    @abstractmethod
    def predict(self, rgb_uint8):
        """Run on a single H x W x 3 uint8 RGB image. Returns a model-specific dict."""

    def per_condition_stats(self, prediction) -> dict:
        """Return per-image stats from a single prediction.

        Floats only — these are averaged across images.
        Typical fields: 'detected', 'n_objects', 'mean_score'.
        """
        return {}

    def vs_reference(self, prediction, reference) -> dict:
        """Per-image metrics comparing prediction to reference (GT-RGB result).

        Floats only. NaN signals "not applicable for this image"
        (e.g. no detections in either) and is excluded from aggregation.
        """
        return {}
