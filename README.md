## Paper
[PointSFDA](https://arxiv.org/abs/2503.15144)

## Environments
Please refer to the environment of [SnowflakeNet](https://github.com/AllenXiangX/SnowflakeNet).

## Pretrained Source Model
[pretrained source model](https://drive.google.com/drive/folders/1t_hSDwtn9TicSW9kZWClXRySKVtxlPHt?usp=drive_link) in CRN datasets

## Train Model
` CUDA_VISIBLE_DEVICES=0 python main.py --config configs/3DFUTURE/SnowflakeNet.yaml `

## Test Model
You need add source_model_path and model_path in the config like:
![2025-03-20 11-16-26屏幕截图](https://github.com/user-attachments/assets/27fe4757-4769-43d2-9f71-05548542fc34)

` CUDA_VISIBLE_DEVICES=0 python main.py --config configs/3DFUTURE/SnowflakeNet.yaml --test`

## Acknowledgements
Some of the code of this repo is borrowed from:
- [SnowflakeNet](https://github.com/AllenXiangX/SnowflakeNet)
- [PoinTr](https://github.com/yuxumin/PoinTr)
- [PCN](https://github.com/wentaoyuan/pcn)
  
