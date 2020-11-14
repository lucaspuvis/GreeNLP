from typing import Dict, List, Union
import carbontracker

from catalyst.core.callback import Callback, CallbackNode, CallbackOrder
from carbontracker.tracker import CarbonTracker


class CarbontrackerCallback(Callback):
    """
    CosineLossCallback
    This callback is calculating cosine loss between hidden states
    of the two hugging face transformers models.
    """

    def __init__(
        self,
        prefix: str = "carbontracker",
        epochs: int = 1,
        **metric_kwargs,
    ):

        super().__init__(
            prefix=prefix,
            **metric_kwargs,
        )

        self.tracker = carbontracker(epochs=epochs)
    
    def on_epoch_start(self, runner: "IRunner"):
        self.tracker.epoch_start()
    

    def on_epoch_end(self, runner: "IRunner"):
        self.tracker.epoch_end()

    
    def on_loader_end(self, runner: "IRunner"):
        self.tracker.stop()