# MF-STN: Matrix Factorization for Spatio-Temporal Neural Networks with Applications to Urban Flow Prediction

This is the MXNet implementation of MF-STN in the following paper:

- Zheyi Pan, Zhaoyuan Wang, Weifeng Wang, Yong Yu, Junbo Zhang, and Yu Zheng. [Matrix Factorization for Spatio-Temporal Neural Networks with Applications to Urban Flow Prediction](https://www.researchgate.net/publication/336847533_Matrix_Factorization_for_Spatio-Temporal_Neural_Networks_with_Applications_to_Urban_Flow_Prediction). 2019. In The 28th ACM International Conference on Information and Knowledge Management (CIKM’19), November 3–7, 2019, Beijing, China.
---

## Requirements for Reproducibility

### System Requirements:

- System: Ubuntu 16.04
- Language: Python 3.5.2
- Devices: a single GTX 1080 GPU


### Library Requirements:

- scipy == 1.2.1
- numpy == 1.16.3
- pandas == 0.24.2
- mxnet-cu90 == 1.5.0
- dgl == 0.2
- tables == 3.5.1
- pymal
- h5py

Dependency can be installed using the following command:

`pip install -r requirements.txt`

---

## Data Preparation

Unzip the data files in:

- `TaxiBJ/data.zip`
- `TaxiNYC/data.zip`

### Description of Flow Data

The flow data are collected from Beijing and New York city. The shape of data are `(date, hour, row, colume, flow_type)`. 

---

## Model Training & Testing

Given TaxiBJ task as example (TaxiNYC task is exactly the same as TaxiBJ task):

1. `cd TaxiBJ/`.
2. The settings of the models are in the folder `src/model_setting`, saved as yaml format. Four models and their enhanced versions by matrix factorization are provided: `cnn`, `gru`, `conv_gru`, and `resnet`.
3. All trained model will be saved in `param/`. There are two types of files in this folder:
   1. `model.yaml`: the model training log (the result on evaluation dataset of each epoch). This file records the number of the best epoch for the model.
   2. `model-xxxx.params`: the saved model parameters of the best evaluation epoch, where `xxxx` is the epoch number.
4. Running the codes:
   1. `cd src/` .
   2. `python train.py --file model_settting/[model_name].yaml --gpus [gpu_ids] --epochs [num_epoch]`. The code will firstly load the best epoch from `params/`, and then train the models for `[num_epoch]`. Our code can be trained with multiple gpus. An example of `[gpu_ids]` is `0,1,2,3` if you have four gpus. But we recommend to use a single gpu to train & evaluate the model if possible. 
5. Training from the begining:
   1. Remove the model records in `param/`, otherwise the code will train from the best pre-trained (saved) model.
   2. Train the model (example).
      1. Single gpu: `python train.py --file model_setting/resnet_mf.yaml --gpus 0 --epochs 200`. 
      2. Multiple gpus: `python train.py --file model_setting/resnet_mf.yaml --gpus 0,1 --epochs 200`.
6. Testing the model (example): `python train.py --file model_setting/resnet_mf.yaml --gpus 0 --epochs 0`. The code will directly give evaluation results on evaluation dataset and test dataset, without training.

---

## Citation

If you find this repository, e.g., the code and the datasets, useful in your research, please cite the following paper:

- Zheyi Pan, Zhaoyuan Wang, Weifeng Wang, Yong Yu, Junbo Zhang, and Yu Zheng. Matrix Factorization for Spatio-Temporal Neural Networks with Applications to Urban Flow Prediction. 2019. In The 28th ACM International Conference on Information and Knowledge Management (CIKM’19), November 3–7, 2019, Beijing, China.

---

## License

MF-STN is released under the MIT License (refer to the LICENSE file for details).