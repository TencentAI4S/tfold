# Install
## Conda[Recommend]
```shell
conda env create -n tfold -f environment.yaml
conda activate tfold
git clone https://github.com/aqlaboratory/openfold.git
cd openfold
pip install .
```
## Pip
### install mmseqs from source
```shell
wget https://github.com/soedinglab/MMseqs2/releases/download/14-7e284/mmseqs-linux-avx2.tar.gz
tar xvfz mmseqs-linux-avx2.tar.gz
echo 'export PATH=$(pwd)/mmseqs/bin/:$PATH' >> ~/.bashrc
source ~/.bashrc
```
### install python packages
```shell
pip install -r requirements.txt
wget https://github.com/aqlaboratory/openfold/archive/refs/tags/v1.0.1.zip -O openfold-1.0.1.zip
unzip openfold-1.0.1.zip
cd openfold-1.0.1 
sed -i 's/deepspeed.utils.is_initialized()/deepspeed.comm.comm.is_initialized()/' openfold/model/primitives.py 
pip install . 
```
