# sklearn-workflows

## Installation

Install and set up [Miniconda3](https://docs.conda.io/en/latest/miniconda.html)

Clone git repository and submodules:

```bash
git clone git@github.com:ruppinlab/sklearn-workflows.git
cd sklearn-workflows
git submodule update --init --recursive
```

To install conda environment on Intel architecture hardware:

```bash
conda env create -f envs/sklearn-workflows-mkl.yml
```

Otherwise:

```bash
conda env create -f envs/sklearn-workflows.yml
```

Activate the environment:

```bash
conda activate sklearn-workflows
```

Add channels to config:

```bash
conda config --env --add channels bioconda
conda config --env --add channels conda-forge
```

Pin fixed package versions/builds:

```bash
echo 'libblas[build=*mkl]' >> "$(conda info --base)/envs/sklearn-workflows/conda-meta/pinned"
echo 'scikit-learn=0.22.1' >> "$(conda info --base)/envs/sklearn-workflows/conda-meta/pinned"
```

Install non-conda packages into environment:

```bash
./utils/install_nonconda_r_pkgs.R
```

## Updates

To update the conda environment on Intel architecture hardware:

```bash
conda env update -f envs/sklearn-workflows-mkl.yml
```

Otherwise:

```bash
conda env update -f envs/sklearn-workflows.yml
```
