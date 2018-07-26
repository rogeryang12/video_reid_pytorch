<<<<<<< HEAD
# video_reid_pytorch
Add `reid` to your `PYTHONPATH`, and you can run examples in `examples` folder.

## Result
Here are some results on `MARS` and `iLIDS` datasets.

### MARS
| Method                                  | rank1 | rank5 | rank10 | mAP   |
|:----------------------------------------|:------|:------|:-------|:------|
| cross entropy                           | 76.11 | 89.55 | 92.63  | 60.11 |
| cross entropy + batch hard triplet      | 80.35 | 91.26 | 93.54  | 67.45 |
| cross entropy + adaptive weight triplet | 80.00 | 92.37 | 94.44  | 69.02 |

### iLIDS
| Method                                  | rank1 | rank5 | rank10 |
|:----------------------------------------|:------|:------|:-------|
| cross entropy                           | 45.82 | 74.77 | 83.89  |
| cross entropy + adaptive weight triplet | 59.22 | 86.78 | 92.50  |
=======
# video_reid_pytorch
>>>>>>> 757466625c339d7da744ec2d89f52406a4d25594
