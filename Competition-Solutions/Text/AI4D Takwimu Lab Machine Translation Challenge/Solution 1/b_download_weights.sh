#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

cd $HOME/solution/models

# 2.2 GB
cd run-20210531-0044-to-mt5b-finetune-fon
echo "Downloading weights. Model 1 of 4..."
curl -L -o model-prefinetuned-fon.bin https://www.dropbox.com/s/h0h7fcqkodq7tfz/model-prefine-jw300-fon-f0-e039-0.5389.bin?dl=0

# 2.2 GB
echo "Downloading weights. Model 2 of 4..."
curl -L -o model-best-f0-e002-0.7313.bin https://www.dropbox.com/s/h62doas4weckvqe/model-best-f0-e002-0.7313.bin?dl=0
cd ..

# 2.2 GB
cd run-20210531-0046-to-mt5b-finetune-ewe
echo "Downloading weights. Model 3 of 4..."
curl -L -o model-prefinetuned-ewe.bin https://www.dropbox.com/s/owcm2xak8you3ar/model-prefine-jw300-ewe-f0-e004-0.3223.bin?dl=0

# 2.2 GB
echo "Downloading weights. Model 4 of 4..."
curl -L -o model-best-f0-e005-0.2852.bin https://www.dropbox.com/s/01dln3sdd4d404j/model-best-f0-e005-0.2852.bin?dl=0

cd ..
cd ..

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


