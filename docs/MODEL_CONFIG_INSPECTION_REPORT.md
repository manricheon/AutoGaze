# External Model Config Inspection Report

| model | status | architecture | model_type | patch_size | crop_or_image_size | frames_per_clip | tubelet_size | hidden_size | rope_indicators | projector_connector_indicators | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| longvila_r1 | inspected | ["VILAForCausalLM"] | vila | 14 | 448 |  |  | 3584 | {"rope_scaling": [null], "rope_theta": [1000000.0]} | {"mm_projector": ["mlp_downsample_2x2_fix"], "mm_projector_cfg": "present_nested", "mm_projector_lr": [null], "mm_projector_type": ["mlp_downsample_2x2_fix"], "projector_hidden_act": ["gelu_fast"], "tune_mm_projector": [true]} | config-only inspection; no weights loaded |
| vjepa2 | inspected | ["VJEPA2Model"] | vjepa2 | 16 | 256 | 64 | 2 | 1024 | {} | {} | config-only inspection; no weights loaded |
