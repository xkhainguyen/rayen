# How to use the ETH cluster Euler

Read [Overview](https://docs.google.com/presentation/d/1y3iSIHqS2lKfDFyogOT1a8iEr3RIxxNjOJfKGM-5LAI/edit#slide=id.g27d9e0ee24_0_266) first

Get access — create account (associated with your ETH email)

```bash
ssh username@euler.ethz.ch
```

Verify your ETH email

Set up SSH keys

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_euler 
ssh-copy-id -i ~/.ssh/id_ed25519_euler.pub username@euler.ethz.ch
```

Request access to MAVT-RSL-HPC. Check members

```bash
getent group MAVT-RSL-HPC
```

After done with cluster access, start installing stuff

```bash
module load gcc/8.2.0 cuda/11.0.3 vim/8.1.1746 eth_proxy gmp/6.1.2 
module save
```

[Workflow](https://bitbucket.org/leggedrobotics/cluster-workflows/src/main/) for more projects

[Jonas](https://github.com/JonasFrey96/ASL_leonhard_euler) shows Python workflow

```bash
conda create -n rayen-env python=3.9.0
conda activate rayen-env
```

Clone RAYEN and install packages

```bash
git clone --recurse-submodules https://github.com/leggedrobotics/rayen.git
cd rayen && pip install -e .
pip install matplotlib
```

Use [Slurm](https://scicomp.ethz.ch/wiki/LSF_to_Slurm_quick_reference) to work with jobs

Interactive job

```bash
srun --pty -n 2 --x11 bash
srun --pty -n 16 --mem-per-cpu=2048 --gpus=1 --gres=gpumem:10240 bash
```
