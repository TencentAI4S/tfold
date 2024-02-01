FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04
# use an older system (18.04) to avoid opencv incompatibility (issue#3524)

# install basics
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
ENV SHELL=/bin/bash
ENV PATH "/miniconda/bin:$PATH"
ENV LD_LIBRARY_PATH "/miniconda/lib:$LD_LIBRARY_PATH"
ENV PATH "~/mmseqs/bin:$PATH"
ENV PATH "/root:$PATH"
ENV TMPDIR_PRFX "/tmp"
RUN apt-get update && apt-get install -y bash
RUN usermod -s /bin/bash root
RUN apt-get update -y \
 && apt-get install -y apt-utils git curl ca-certificates bzip2 cmake tree htop bmon iotop g++ wget openssh-client openssh-server \
 && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev zip

# Install Miniconda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-py38_4.11.0-Linux-x86_64.sh -O ./miniconda.sh  \
 && chmod +x ./miniconda.sh \
 && bash ./miniconda.sh -b -p ./miniconda \
 && rm ./miniconda.sh

RUN wget  https://github.com/soedinglab/MMseqs2/releases/download/14-7e284/mmseqs-linux-avx2.tar.gz -O ~/mmseqs-linux-avx2.tar.gz \
 && tar xvfz ~/mmseqs-linux-avx2.tar.gz \
    && rm ~/mmseqs-linux-avx2.tar.gz

RUN conda install -y --no-update-deps -c anaconda git
RUN conda install -y --no-update-deps -c pytorch -c nvidia \
        pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda==11.7 \
    && conda install -y --no-update-deps -c pyg pyg pytorch-scatter \
    && conda install -y --no-update-deps -c conda-forge openmm pdbfixer libgcc libstdcxx-ng \
    && conda clean -y --all
# install Python dependencies via conda
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        deepspeed==0.12.3 \
        termcolor==2.3.0 \
        biopython==1.79 \
        ml-collections==0.1.1 \
        dm-tree==0.1.8 \
        modelcif==0.9  \
        numpy==1.21.2 \
        scipy \
    && pip cache purge

RUN wget https://github.com/aqlaboratory/openfold/archive/refs/tags/v1.0.1.zip -O openfold-1.0.1.zip \
    && unzip openfold-1.0.1.zip \
    && cd openfold-1.0.1 \
    && sed -i 's/deepspeed.utils.is_initialized()/deepspeed.comm.comm.is_initialized()/' openfold/model/primitives.py \
    && pip install . \
    && cd .. \
    && rm -rf openfold

RUN git clone https://github.com/TencentAI4S/tfold.git ~/tfold

WORKDIR /root/tfold
CMD [ "/bin/bash" ]

