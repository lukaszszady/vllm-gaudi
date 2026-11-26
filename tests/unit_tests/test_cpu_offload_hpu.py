import pytest
import torch
from vllm.config import VllmConfig, ModelConfig, OffloadConfig, UVAOffloadConfig
from vllm_gaudi.v1.worker.hpu_model_runner import HPUModelRunner
from vllm.model_executor.model_loader import get_model

def test_cpu_offload_param_placement():
    # Setup config with CPU offloading
    offload_config = OffloadConfig(uva=UVAOffloadConfig(cpu_offload_gb=1.0))
    vllm_config = VllmConfig(
        model=ModelConfig(model="facebook/opt-125m"),
        offload_config=offload_config
    )
    
    # Initialize runner
    runner = HPUModelRunner(vllm_config=vllm_config)
    
    # This part requires HPU to actually load the model and run the offloader.
    # On CPU, we can only verify that the runner handles the config.
    # Since we are on CPU, this test will fail during model loading.
    # But the manager wants a test for the HPU CI.
    
    runner.load_model()
    
    # Assert that some parameters are on CPU and some on HPU
    params = list(runner.model.parameters())
    has_cpu = any(p.device.type == "cpu" for p in params)
    has_hpu = any(p.device.type == "hpu" for p in params)
    
    assert has_cpu, "Some parameters should be offloaded to CPU"
    assert has_hpu, "Some parameters should remain on HPU"

