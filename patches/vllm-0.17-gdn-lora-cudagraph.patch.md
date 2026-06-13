# vllm 0.17.0 patch — apply to vllm/lora/model_manager.py (LoRAModelManager.__init__,
# right after `self.supported_lora_modules = get_supported_lora_modules(self.model)`).
# Workaround for https://github.com/vllm-project/vllm/issues/36372:
# Qwen3.5 hybrid-GDN projections (in_proj_qkvz/in_proj_ba) declare 2 subloras in
# packed_modules_mapping but expose 4 fused output slices, so dummy-LoRA warmup during
# CUDA-graph capture crashes (IndexError in set_lora). Excluding GDN modules from LoRA
# wrapping is correct for adapters that target only LM standard modules
# (q/k/v/o/gate/up/down_proj): GDN never carries LoRA at runtime, so capture == runtime.

        _gdn_lora_blocklist = {"in_proj_qkvz", "in_proj_ba", "conv1d", "out_proj"}
        self.supported_lora_modules = [
            m for m in self.supported_lora_modules if m not in _gdn_lora_blocklist
        ]
