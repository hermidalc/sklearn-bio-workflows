# sklearn-bio-workflows

## Installation

Install and set up [Miniforge3](https://github.com/conda-forge/miniforge#download)

Clone git repository and submodules:

```bash
git clone --recurse-submodules https://github.com/hermidalc/sklearn-bio-workflows.git
cd sklearn-bio-workflows
```

To install conda environment on Intel architecture hardware:

```bash
mamba env create -f envs/sklearn-bio-workflows-mkl.yml
```

Otherwise:

```bash
mamba env create -f envs/sklearn-bio-workflows.yml
```

Activate the environment:

```bash
mamba activate sklearn-bio-workflows
```

Install non-conda packages into environment:

```bash
./utils/install_nonconda_r_pkgs.R
```

## Updates

Update the git repository and submodules:

```bash
git pull
git submodule update --recursive
```

To update the conda environment on Intel architecture hardware:

```bash
mamba env update -f envs/sklearn-bio-workflows-mkl.yml
```

Otherwise:

```bash
mamba env update -f envs/sklearn-bio-workflows.yml
```
