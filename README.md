# ECG-data-augmentation-for-training-deep-learning-classifiers
Scripts and modules for training and testing deep neural networks with data augmentation (DA) for ECG classification.
Repository to bachelor thesis 'ECG Analysis Using Deep Learning: Exploring Data Augmentations' Effect on Model Performance' done at KTH Royal Institue of Technology.

Bibtex:
```
@mastersthesis{ecg_da_exploration_2025,
  title = {ECG Analysis Using Deep Learning: Exploring Data Augmentations' Effect on Model Performance},
  author = {Kent Tieu and Amjad Alakrami},
  type = {Bachelor Thesis}
  year = {2025},
  school={KTH, School of Electrical Engineering and Computer Science (EECS)},
  address = {Stockholm, Sweden},
  series = {TRITA-EECS-EX},
  number = {2025:149}
}
```
-----

## Requirements
This code was tested on Python 3.12 with Tensorflow `2.18`. 

## Model
The model used in the paper is a residual neural. The neural network architecture implementation in Keras is available in ``model.py``. To print a summary of the model layers run:
```bash
$ python model.py
```

![resnet](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41467-020-15432-4/MediaObjects/41467_2020_15432_Fig3_HTML.png?as=webp)

The model receives an input tensor with dimension `(N, 5000, 12)`, and returns an output tensor with dimension `(N, 5)`,
for which `N` is the batch size.

The model can be trained using the script `train.py`. 

- **input**: `shape = (N, 5000, 12)`. The input tensor should contain the  `5000` points of the ECG tracings
sampled at `500Hz` (i.e., a signal of approximately 10 seconds). The last dimension of the 
tensor contains points of the 12 different leads. The leads are ordered in the following order: 
`{I, II, III, aVL, aVR, aVF, V1-V6}`.


- **output**: `shape = (N, 5)`. Each entry contains a probability between 0 and 1, and can be understood as the
probability of a given abnormality to be present. The abnormalities it predicts are  **(in that order)**: (NORM), (CD), (HYP), (MI), (STTC). The abnormalities are not mutually exclusive, so the probabilities do not necessarily
sum to one.

![abnormalities](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41467-020-15432-4/MediaObjects/41467_2020_15432_Fig1_HTML.png?as=webp)

## Datasets
The dataset used in the paper is from PTB-XL, a large publicly available electrocardiography dataset, 
and can be downloaded in [doi:https://doi.org/10.1038/s41597-020-0495-6](https://doi.org/10.1038/s41597-020-0495-6)

## Scripts

- ``train.py``: Script for training the neural network. To train the neural network run: 
```bash
$ python train.py PATH_TO_TRAINSET_HDF5 PATH_TO_TRAIN_LABELS_CSV PATH_TP_VALIDSET_HDFT PATH_TO_VALID_LABELS_CSV --final_model_name "MODEL_NAME" [options]
$ python train.py PATH_TO_TRAINSET_HDF5 PATH_TO_TRAIN_LABELS_CSV PATH_TP_VALIDSET_HDFT PATH_TO_VALID_LABELS_CSV \
  --final_model_name myModelName \
  --DA "DA_NAME" "add_gaussian_noise" \
  --BS 64 \ 
  --Balanced True
```
More options are available, see ``train.py`` for all options.

Pre-trained models obtained using such script can be found in `./trained_models`

To evaluate model performance:
- ``evalPrediction.py``: Script for generating figures and tables used for evaluating model performance.
```bash
$ python evalPrediction.py PATH_TO_TESTSET_HDF5 PATH_TO_TEST_LABELS_CSV PATH_TO_OUTPUT PATH_TO_MODEL
```

If you only want to generate predictions, use the following:
- ``predict.py``: Script for generating the neural network predictions on a given dataset. It saves as a `.npy` file at specified path.
```bash
$ python predict.py PATH_TO_TESTSET_HDF5 PATH_TO_MODEL_KERAS --ouput_file PATH_TO_OUTPUT_FILE 
```

- ``model.py``: Auxiliary module that defines the architecture of the deep neural network.
To print a summary of the model  layers run:
```bash
$ python model.py
```

## Helper Scripts
A set of helper scripts is provided in `./helping_tools` to be used with the dataset from PTB-XL, that generates the necessary HDF5 and CSV files from the ECG signals from PTB-XL. The scripts include some pre-processing that removes samples not belonging to any of the diagnosis. Replace necessary paths.

First, use the ``csv_maker.py`` to create the csv files. Uncomment the `print(idex)` and run:
```bash
$ python csv_maker.py
```
Copy the output and open ``modifyRecord.py`` and paste the output into `inv_idex`.  This tracks what samples to be filtered. Run ``modifyRecord.py``:
```bash
$ python modifyRecord.py
```
and lastly run:
```bash
$ python to_hdf5.py
```
The necessary pre-processing should be done.

## Acknowledgements
This repository's residual model is based on the model described in the article [Automatic diagnosis of the 12-lead ECG using a deep neural network](https://doi.org/10.1038/s41467-020-15432-4). The model and predict code is based on the repo belonging to the article[Automatic ECG diagnosis using a deep neural network](https://github.com/antonior92/automatic-ecg-diagnosis), and modified to suit the new usecase.

The original repository is licensed under the [MIT License](https://opensource.org/licenses/MIT), which permits modification and redistribution.

