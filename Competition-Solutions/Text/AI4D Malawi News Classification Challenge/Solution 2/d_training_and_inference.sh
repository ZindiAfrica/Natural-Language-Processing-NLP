#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

cd $HOME/solution\ 2/models

cd run-20210508-1501-tf-mt5large-len144-b24-7e5-4v100
echo "Training model $(pwd)..."
python3 run.py
cd ..

cd run-20210508-1646-tf-mt5large-len256-b24-7e5-4v100
echo "Training model $(pwd)..."
python3 run.py
cd ..

cd run-20210509-1534-tf-mt5large-len64-b24-7e5-4v100
echo "Training model $(pwd)..."
python3 run.py
cd ..

cd run-20210509-1605-tf-mt5large-len128-b24-7e5-4p40
echo "Training model $(pwd)..."
python3 run.py
cd ..

cd run-20210509-1614-tf-mt5large-len192-b24-7e5-4v100
echo "Training model $(pwd)..."
python3 run.py
cd ..

cd run-20210509-1735-tf-mt5xl-len128-b16-7e5-1a100
echo "Training model $(pwd)..."
python3 run.py
cd ..
cd ..

python3 ensemble.py

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


