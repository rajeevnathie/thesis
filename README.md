README

This code is written for my thesis using data from a picture naming task.

Three models were chosen an ran for inference on the data.
'M-CLIP/XLM-Roberta-Large-Vit-B-32' - CLIP VLM
'xlm-roberta-base' - RoBerta Transformer
Dutch ELMo model acquired from: https://github.com/HIT-SCIR/ELMoForManyLangs (more information on how to set up this model can be found here as well)

Make sure to either setup an environment (using conda - optional) and load the requirements.txt file or ensure that all packages are installed from the requirements.txt file.

CHANGES TO BE MADE:
Model validation files:
model.py:
- image_folder: set path to absolute or relative path of the image folder you want to use
elmo.py:
- model_path: set path to absolute or relative path of downloaded ELMo model that should be used
clip_model.py:
- image_folder: set path to absolute or relative path of the image folder you want to use
- labels: change list to labels that should be used for your task
alternative_model.py
- image_folder: set path to absolute or relative path of the image folder you want to use
- labels: change list to labels that should be used for your task

Model code for inference on task data:
results.py:
- elmo_model_path: set path to absolute or relative path of downloaded ELMo model that should be used
- changed_data_pairs: this variable was introduced to fix errors in data files, can be removed if not needed anymore.
- line 143: for loop iterating local directory of data csvs. Change to local directory (same for line 148)
- line 233: change path to folder where new data has to be stored

Setup:
Clone repository to your device and make sure to follow each step listed above. 
Make sure to correctly clone the Dutch ELMo model after this as well into the project.
For model validation run files: clip_model.py for CLIP model. alternative_model.py for CLIP model using different image and text encoder. elmo.py for elmo model validation.
Run process_csv.py to go over data and create similarity matrices of the available data.
create_plots.py creates scatter plots for average RT of participants
results_csv.py runs models and appends data to csv and stores csvs in a different directory.

