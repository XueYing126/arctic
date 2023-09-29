# DIGIT model on  ARCTIC

## Project Overview
This project is developed based on [ARCTIC](https://github.com/zc-alexfan/arctic).  Please follow the [ARCTIC instructions](https://github.com/zc-alexfan/arctic#arctic--a-dataset-for-dexterous-bimanual-hand-object-manipulation) to set up the environment and download the ARCTIC dataset.

We have implemented a method based on [DIGIT](https://github.com/zc-alexfan/digit-interacting), which utilizes per-pixel part segmentation masks to supervise the learning process. 
We made modifications to the pose estimator. Instead of using the 2.5D representation and regressing the hand joint heatmaps, we directly regress MANO parameters, similar to the decoder in ARCTIC_SF. For more details on the implmentation, please refer to this [report]().

This approach effectively addresses the consistent motion reconstruction task.



## Obtain Segmentation Mask

To obtain the part segmentation mask for ARCTIC data, please follow the instructions in the [render_mano_ih README.md](https://github.com/XueYing126/render_mano_ih).

## Train
The following code trains the single-frame allocentric DIGIT model with HRNet as backbone. 
```bash
# in the allocentric setting, use HRNet as backbone
python scripts_method/train.py --setup p1 --method digit_hrnet --trainsplit train --valsplit minival 
```

The following code trains the single-frame allocentric DIGIT model with ResNet50 as backbone. 
```bash
# in the allocentric setting, use ResNet50 as backbone
python scripts_method/train.py --setup p1 --method digit_resnet50 --trainsplit train --valsplit minival 
```
Please note that the model has around 60M parameters. If you have encountered GPU out of memory issue, you can add `--batch_size=32`  to the end of the command.

## Evaluation

### Download the pre-trained model:
We have trained weights for HRNet/ResNet50 backbones and p1/p2 splits. Please download the trained weights from this [link]().





## Test and Submit to [Leaderboard](docs/leaderboard.md)

Replace `PATH_TO_WEIGHTS` with the path to weights. e.g. logs/5d91ff741/checkpoints/last.ckpt

Replace `METHOD` with the method name: `digit_hrnet` for DIGIT with HRNet as backbone, `digit_resnet50` for DIGIT with resnet50 as backbone.


```bash
python scripts_method/extract_predicts.py --setup p1 --method METHOD --load_ckpt PATH_TO_WEIGHTS --run_on test --extraction_mode submit_pose
```



## Citation
```bibtex
@inproceedings{fan2023arctic,
  title = {{ARCTIC}: A Dataset for Dexterous Bimanual Hand-Object Manipulation},
  author = {Fan, Zicong and Taheri, Omid and Tzionas, Dimitrios and Kocabas, Muhammed and Kaufmann, Manuel and Black, Michael J. and Hilliges, Otmar},
  booktitle = {Proceedings IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2023}
}

@inProceedings{fan2021digit,
  title={Learning to Disambiguate Strongly Interacting Hands via Probabilistic Per-pixel Part Segmentation},
  author={Fan, Zicong and Spurr, Adrian and Kocabas, Muhammed and Tang, Siyu and Black, Michael and Hilliges, Otmar},
  booktitle={International Conference on 3D Vision (3DV)},
  year={2021}
}
```