#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

cd $HOME/solution/models

cd run-20210529-1644-to-mt5b-prefinetune-jw300-fon
echo "Pre-finetuning model $(pwd)..."
python3 run.py
cd ..

cd run-20210529-1647-to-mt5b-prefinetune-jw300-ewe
echo "Pre-finetuning model $(pwd)..."
python3 run.py
cd ..

# Copy pre-finetuned weights
mv run-20210529-1644-to-mt5b-prefinetune-jw300-fon/*.bin run-20210531-0044-to-mt5b-finetune-fon/model-prefinetuned-fon.bin
mv run-20210529-1647-to-mt5b-prefinetune-jw300-ewe/*.bin run-20210531-0046-to-mt5b-finetune-ewe/model-prefinetuned-ewe.bin

cd run-20210531-0044-to-mt5b-finetune-fon
echo "Finetuning model $(pwd)..."
python3 run.py
cd ..

cd run-20210531-0046-to-mt5b-finetune-ewe
echo "Finetuning model $(pwd)..."
python3 run.py
cd ..
cd ..

python3 join_submission_fon_ewe.py

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


