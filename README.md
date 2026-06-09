# DinoGRL
Official PyTorch implementation of "DINOv2 Driven Gait Representation Learning for Video-Based Visible-Infrared Person Re-Identification" (ACM MM 2025).



## Model Zoo

Please download the pretrained checkpoints from the following links and place them under `checkpoints/`.

| Dataset   | File              | Download                                                                                    |
| --------- | ----------------- | ------------------------------------------------------------------------------------------- |
| HITSZ-VCM | `best_model.pth`  | [Download](https://github.com/yyj-001/DinoGRL/releases/download/HITSZ-VCM/best_model.pth)  |
| HITSZ-VCM | `best_model2.pth` | [Download](https://github.com/yyj-001/DinoGRL/releases/download/HITSZ-VCM/best_model2.pth) |
| HITSZ-VCM | `best_shape.pth`  | [Download](https://github.com/yyj-001/DinoGRL/releases/download/HITSZ-VCM/best_shape.pth)  |

| Dataset     | File              | Download                                                                                    |
| ----------- | ----------------- | ------------------------------------------------------------------------------------------- |
| BUPT-Campus | `best_model.pth`  | [Download](https://github.com/yyj-001/DinoGRL/releases/download/BUPT/best_model.pth)  |
| BUPT-Campus | `best_model2.pth` | [Download](https://github.com/yyj-001/DinoGRL/releases/download/BUPT/best_model2.pth) |
| BUPT-Campus | `best_shape.pth`  | [Download](https://github.com/yyj-001/DinoGRL/releases/download/BUPT/best_shape.pth)  |

After downloading, please organize the checkpoints as follows:

```text
checkpoints/
├── best_model.pth
├── best_model2.pth
└── best_shape.pth
```

## Acknowledgements
Our code uses `MaskBranch_vits14.pt` and `dinov2_vits14_pretrain.pth` from BigGait. We sincerely thank the authors for their excellent work. Please refer to their official GitHub repository: [OpenGait](https://github.com/ShiqiYu/OpenGait).
