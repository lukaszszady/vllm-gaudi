import torch

import vllm.model_executor.model_loader.utils as hpu_utils
import vllm.model_executor.model_loader.base_loader as base_loader

_upstream_process_weights_after_loading = hpu_utils.process_weights_after_loading


def hpu_process_weights_after_loading(model, model_config, target_device):
    """Gaudi override: accept device strings (e.g., "hpu")."""
    _upstream_process_weights_after_loading(model, model_config, torch.device(target_device))


hpu_utils.process_weights_after_loading = hpu_process_weights_after_loading
base_loader.process_weights_after_loading = hpu_process_weights_after_loading
