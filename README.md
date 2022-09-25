# CNNStack

## 1. Project structure

In the root folder, you will find the main files used during the experiment. Those are `train.py` (for training single models), `test.py` (for evaluating on the test set), and `ensemble.py` (for creating and evaluating ensembles).

- :file_folder: `./utils/`

The **utils** folder contains the preprocessing script used for the dataset. 

- :file_folder: `./models/`

The **models** folder contains the models used in this experiment. 


## 2. Dataset structure
Each dataset should be structured as follows:
- :file_folder: `./dataset/`
  - :file_folder: `generated/`
  - :file_folder: `real/`
## 3. Training models
`python3 train.py --model model_name --train_fold train_folder --val_fold val_folder --batch bs --lr lr --epochs n_epochs`
## 4. Training ensemble
`python3 ensemble.py --train_fold training_folder --val_fold validation_folder`
## 5. Testing
`python3 test.py --test_fold test_folder`
