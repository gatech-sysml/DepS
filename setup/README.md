### Setup conda environment on your node

- Install OpenMPI 4.1.2 and set PATH and LD_LIBRARY_PATH accordingly
- Create conda env from env.yaml `conda env create -f env.yaml`
- Activate environment `conda activate deps`
- Install horovod `pip install horovod==0.19.5`
- Check horovod installation with `horovodrun --check-build`
- Setup `WANDB_DIR` by `export WANDB_DIR=<SOME_PATH>`

### Conda Clean Cache
```
conda clean --all
```

### Conda Remove Env
```
conda remove --all --name <env_name>
```
