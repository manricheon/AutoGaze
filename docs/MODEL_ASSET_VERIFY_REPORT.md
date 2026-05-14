# External Model Asset Verification Report

| model | download_status | local_exists | config_ok | processor_tokenizer_ok | weights_ok | config_example_exists | adapter_resolves | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| longvila_r1 | missing | False | False | False | False | True | True | local directory missing; missing config files: ['config.json']; missing processor/tokenizer files: ['preprocessor_config.json', 'tokenizer_config.json']; weights missing or incomplete |
| vjepa2 | local_exists | True | True | True | True | True | True | local files verified without loading model weights |
