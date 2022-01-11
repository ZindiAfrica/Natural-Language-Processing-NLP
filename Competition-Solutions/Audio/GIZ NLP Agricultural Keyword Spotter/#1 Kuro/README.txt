## Organisation of the folder :

agrinet
    │  
    ├── merge_csv.py    <-- merge the 4 singles models trained WITHOUT pseudo labels
	│
	├── resample.py  	<-- resample audio wav files provided by zindi. audio_files into audio_files-48000 ; AdditionalUtterances into AdditionalUtterances-48000 ; nlp_keywords into nlp_keywords-48000
	│
	├── merge_csv-PL.py <-- merge the 2 models trained with pseudo labels
	│
	├── model_1_cleaned.py <-- single model 1
	├── model_4_cleaned.py <-- single model 4
	├── model_6_cleaned.py <-- single model 5
	├── model_7_cleaned.py <-- single model 7
	│
	│
	├── model_pl_A_cleaned.py <-- single model trained with the pseudo label from the blending of the 4 previous models.
	├── model_pl_B_cleaned.py <-- single model trained with the pseudo label from the blending of the 4 previous models.
    │
    │        					
    ├── audio_files       						<-- First audio wav files provided by zindi.
    ├── AdditionalUtterances/latest_keywords    <-- Second audio wav files provided by zindi.
    ├── nlp_keywords      						<-- Third audio wav files provided by zindi.
    │  
    ├── audio_files-48000      						  <-- First audio wav files provided by zindi, resampled at 48 kHz.
    ├── AdditionalUtterances-48000/latest_keywords    <-- Second audio wav files provided by zindi, resampled at 48 kHz.
    ├── nlp_keywords-48000      					  <-- Third audio wav files provided by zindi, resampled at 48 kHz.
	
	

## Resampling audio files
## In our training, we resample all our file in 48 kHz, in a uncompressed format in order to load them faster with torchlibrosa. You can download the data on ggdrive  or recreate them using the script resample.py
## link https://drive.google.com/drive/folders/1U7-pSD24xai9ecYKb7f-BgMANG-VDBCM?usp=sharing : files to download  audio_files-48000.zip, nlp_keywords-48000.zip and AdditionalUtterances-48000.zip

python resample.py


## Commands to train the 4 single models (EfficientNet Architectures):

python model_1_cleaned.py
python model_4_cleaned.py
python model_6_cleaned.py
python model_7_cleaned.py

## Commands to merge the prediction of the 4 single models (average). The blending obtained should already give the 1st place in the LB : 

python merge_csv.py

## We can improve our model using the prediction of the test [submission_ensemblingv5.csv] (pseudo labels) to increase the number of samples during our training. This methods is a semi-supervised learning approach. 
## Each of them should also give the 1st place on the Leaderboard. We trained two models :

python model_pl_A_cleaned.py
python model_pl_B_cleaned.py

## Finally, we can merge (average) our two models which use the semi supervised-learning approach, which give a boost compare to our previous blending. The final filename should be submission_ensemblingv7-PL.csv
python merge_csv-PL.py