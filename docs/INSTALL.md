# Install
## Conda[Recommend]
```shell
conda env create -n tfold -f environment.yaml
conda activate tfold
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
```
