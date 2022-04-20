# pzflow-paper-2021

Plots for the pzflow 2021 paper

To install packages for this repo:

1. Run `conda env create -f environment.yml`
2. Activate the new environment via `conda activate pzflow-paper`
3. Run `poetry install`
4. If you want to use a GPU using Cuda, run

```shell
pip install --upgrade "jax[cuda]==0.3.7" -f https://storage.googleapis.com/jax-releases/jax_releases.html
```
