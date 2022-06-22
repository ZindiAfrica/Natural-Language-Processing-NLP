#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

cd $HOME/solution/models

cd run-20210531-0044-to-mt5b-finetune-fon
echo "Inference from model $(pwd)..."
python3 run.py --job=test
cd ..

cd run-20210531-0046-to-mt5b-finetune-ewe
echo "Inference from model $(pwd)..."
python3 run.py --job=test
cd ..
cd ..

python3 join_submission_fon_ewe.py

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


