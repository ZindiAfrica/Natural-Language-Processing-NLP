#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

# Extracted solution directory expected to be at $HOME path

cd $HOME/solution
sudo apt-get update
sudo apt-get -y install python3-pip unzip
pip3 install --upgrade pip
pip3 install -r requirements.txt

# Install CUDA (11.1) and cuDNN (8.0.4)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
