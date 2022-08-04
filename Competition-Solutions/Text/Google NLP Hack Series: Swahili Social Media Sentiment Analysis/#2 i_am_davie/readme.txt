
---------------------------Requirements-----------------------------------------

-GPU enabled environment (preferably kaggle or google colab)
-pandas 
-numpy 
-matplotlib
-tqdm
-sklearn
-catboost
- easynmt

Note: All above packages are preinstalled on both google colab and kaggle except:

- easynmt ( install using !pip install -U easynmt)

Why easynmt?
This library provides an easy to use wrapper around M2M models which where used for translation purposes!
There




------------------------------ Instructions --------------------------------------

First we need to specify a path to project directory where we will find datasets:
 1. Train.csv
 2. Test.csv
 3. sample_submission.csv 
 

In this notebook i store path of said directory on the variable `proj_dir`
Review the first few cells to set this path appropriately!!!

Note: If running on google colab and the project directory files are stored in google drive:
******you will need to authenticate appropriately for mounting****************

Just run all the cells to produce the outputs

------------------------------ Outputs ----------------------------------------

Running the notebook should produce two .csv files in the working directory:
 1. finalSubmission.csv
 
 The submission file to be considered is finalSubmission.csv
 
------------------------------- Recommendations ----------------------------------------
 
Better translations to English were possible through google translate python API but it was replaced by M2M
model because of the request limits when using google translate API


