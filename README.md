# CelebAMask-HQ with fcn_resnet50


### Model Training
*Note* i have saved 4 prediction images to see if model is really learning it is not clearly visible segmentaion,created just for see how model is performing.

I have Trained model on Nvidia RTX 2070 GPU using horovod docker for refer <a href="https://github.com/horovod/horovod">link</a>.
but below i have mention commands to install HOROVOD just in case.


Create label by running script using below command 
1. `python create_labels.py --img-dir {Location of CelebA-HQ-img folder} --save-label-dir {Location to save labels folder}`

Run training script by command 
1. `horovodrun -np {num_of_gpu} python horovod_train.py --dataset-dir {Location of CelebA-HQ-img folder} --label-dir {Location of saved labels folder}`.


Onnx model will be stored in *./models/onnx/* directory by default or to change the model directory run the above command with `--onnx-model-dir {location}`.



To install OpenMPI 
```
sudo -i
mkdir /tmp/openmpi && \
cd /tmp/openmpi && \
wget https://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-4.0.0.tar.gz && \
tar zxf openmpi-4.0.0.tar.gz && \
cd openmpi-4.0.0 && \
./configure --enable-orterun-prefix-by-default && \
make -j $(nproc) all && \
make install && \
ldconfig && \
rm -rf /tmp/openmpi
```

To install pytorch with cuda 10.2
1. `pip install torch torchvision`



To install Horovod with Pytorch
1. `HOROVOD_WITHOUT_GLOO=1 HOROVOD_WITHOUT_TENSORFLOW=1 HOROVOD_WITHOUT_MXNET=1 HOROVOD_WITH_MPI=1 HOROVOD_WITH_PYTORCH=1 python -m pip install horovod[pytorch]`



### Inference

I have shown inference using ONNX-RUNTIME and Pytorch.It is impplemented in *inference.ipynb*.

Models are upload in Gdrive below are the link to download the model.
1. Pytorch model [checkpoint-1.pth.tar](https://drive.google.com/file/d/1OrhIEVdYOd3TvOzao3a4Pz6AgBNlbnQD/view?usp=sharing)
2. Onnx model [checkpoint-onnx-1.onnx](https://drive.google.com/file/d/17m_jhdMGAjm8ieL5Gh61f68zTyD-OYQF/view?usp=sharing)
