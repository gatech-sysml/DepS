## Installing MPI

1. Download MPI (for `DepS` use 4.1.2)

```
wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.2.tar.gz
```

2. Untar the source `tar -xvzf openmpi-4.1.2.tar.gz`

### Installation

1. Configure MPI with relevant parameters
    a. Configure to a path on serenity (shared) and build CUDA aware MPI

```
./configure --prefix=<prefix path> --with-cuda=<CUDA_PATH>
```

2. Install MPI `make all; make install`

3. Check Installation

- `which mpicc`, `which mpirun` - To check installation paths
- `mpirun --version`  - To check MPI version
- `mpirun -np 8 -H <host1>:4,<host2>:4 hostname` - To check multi node MPI
- `echo $PATH`, `echo $LD_LIBRARY_PATH` - To check that correct MPI paths are included. If they are not, update them in `.bashrc`, `.zshrc`
