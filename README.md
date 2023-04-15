# DistHD

This is the source code for paper "DistHD: A Learner-Aware Dynamic Encoding Method for Hyperdimensional Classification". 

## Test Environment
- Python: 3.8.10
- Torch: 1.13.1+cu117
- Numpy: 1.23.5

## Getting Started

### Dataset
We run DistHD on the following dataset: 
- [MNIST](https://www.openml.org/search?type=data&sort=runs&id=554&status=active) 
- [UCIHAR](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones) 
- [ISOLET](https://archive.ics.uci.edu/ml/datasets/isolet)
- [PAMAP2](https://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring)
- [DIABETE](https://archive.ics.uci.edu/ml/datasets/diabetes) 

To try these dataset, download them using links above and store them in `data/NAME.py`. (NAME = Name of the dataset)

### DistHD usage

To execute the code, please preprocess your dataset with `data_preprocess.py` and then run the corresponding main function in `DistHD.py`. 

The following code generates dummy data and trains a DistHD classification model with it. 
The result provided by `model.test()` includes training accuracy and inference accuracy. 
```Python
dim = 10000
n_samples = 1000
n_features = 100
n_classes = 5
learning_rate = 0.35
epochs = 10
nRegen = 20
x = torch.randn(n_samples, features) # dummy data
y = torch.randint(0, classes, [n_samples]) # dummy data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x , y)
model = DistHD(dim, n_features, n_classes, learning_rate)
model.train(x_train, y_train, epochs, nRegen)
model.test(x_train, y_train, x_test, y_test)
```
The following code generates top1, top2, top3 accuracy using baselineHD: 
```Python
baseline = DistHD(dim, n_features, n_classes, learning_rate)
baseline_result = DistHD.compare(x, y, epochs)
```

## Citation Request

If you find the code useful, please cite the following paper: 

Junyao Wang, Sitao Huang, Mohsen Imani  "DistHD: A Learner-Aware Dynamic Encoding Method for Hyperdimensional Classification", 
IEEE/ACM Design Automation Conference (DAC), 2023.

```bibtex
@inproceedings{wang2023disthd,
  title={DistHD: A Learner-Aware Dynamic Encoding Method for Hyperdimensional Classification},
  author={Wang, Junyao and Huang, Sitao and Imani, Mohsen},
  booktitle={Proceedings of the 60th Annual Design Automation Conference},
  year={2023}
}
