# DIGIT model on  ARCTIC

We have implemented a model based on [DIGIT](https://github.com/zc-alexfan/digit-interacting), which utilizes per-pixel part segmentation masks to supervise the learning process. 
We made modifications to the pose estimator. Instead of using the 2.5D representation and regressing the hand joint heatmaps, we directly regress MANO parameters, similar to the decoder in ARCTIC_SF.

This approach effectively addresses the consistent motion reconstruction task.



## Obtain Segmentation Mask

To obtain the part segmentation mask for ARCTIC data, please follow the instructions in the [render_mano_ih README.md](https://github.com/XueYing126/render_mano_ih).

## Train

```bash
python scripts_method/train.py --setup p1 --method arctic_digit_mano --trainsplit train --valsplit minival 
```
Please note that the model has around 60M parameters. If you have encountered GPU out of memory issue, you can add `--batch_size=32`  to the end of the command.

## Evaluate and Submit to [Leaderboard](docs/leaderboard.md)

```bash
python scripts_method/extract_predicts.py --setup p1 --method arctic_digit_mano --load_ckpt logs/5d91ff741/checkpoints/last.ckpt --run_on test --extraction_mode submit_pose
```



## Citation
```bibtex
@inProceedings{fan2021digit,
  title={Learning to Disambiguate Strongly Interacting Hands via Probabilistic Per-pixel Part Segmentation},
  author={Fan, Zicong and Spurr, Adrian and Kocabas, Muhammed and Tang, Siyu and Black, Michael and Hilliges, Otmar},
  booktitle={International Conference on 3D Vision (3DV)},
  year={2021}
}
```