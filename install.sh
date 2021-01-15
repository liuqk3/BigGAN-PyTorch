conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
conda install tqdm numpy scipy h5py

conda install -c anaconda tensorflow-gpu==1.3
# for TFHub
pip install tensorflow-hub==0.2
pip install tensorflow==1.7 # only tensorflow>=1.7 can use hub 