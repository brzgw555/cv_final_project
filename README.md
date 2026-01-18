# cv_final_project

## Requirements

The usual installation steps involve the following commands, they should set up the correct CUDA version and all the python packages

```
conda env create -f environment.yml
conda activate stylegan3
```

Then install the additional requirements

```
pip install -r requirements.txt
pip install -r pips/requirements.txt
```

这是调用RAFT的依赖包，执行上述命令后一般已经满足，若缺失再执行
```
conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.1 matplotlib tensorboard scipy opencv -c pytorch
```

## weights

get stygan2 weights
run

```
python scripts/download_model.py
```

get RAFT weights from[google drive](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT)
并放在.RAFT/models/下

get PIPS weights from[Hugging Face](https://huggingface.co/aharley/pips/tree/main)
模型名为model-000200000.pth,放在 ./pips/checkpoint/下

## run

```
.\scripts\gui.bat
```