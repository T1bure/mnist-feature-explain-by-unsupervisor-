# Environment and dependency requirements

## Tested environment

| Component | Tested value |
|---|---|
| Operating system | Windows |
| Python | 3.10 |
| GPU | NVIDIA GeForce RTX 5070 Ti |
| PyTorch | 2.9.0+cu130 |
| CUDA runtime bundled with PyTorch | 13.0 |
| scikit-learn | 1.7.2 |
| SciPy | 1.15.3 |
| Matplotlib | 3.10.9 |

GPU is optional. Use `--force-cpu` to run the complete experiment without CUDA.

## Python packages

- `torch`
- `torchvision`
- `numpy`
- `scipy`
- `scikit-learn`
- `matplotlib`

Install a suitable PyTorch build by following the [official installation guide](https://pytorch.org/get-started/locally/), then run:

```bash
pip install -r requirements.txt
```

If PyTorch already works in the environment, install only the analysis dependencies:

```bash
pip install scipy scikit-learn matplotlib
```

## Data and disk usage

- MNIST is downloaded automatically on the first run.
- Raw and compressed MNIST files use approximately 65 MB.
- The best model checkpoint uses approximately 3.6 MB.
- Generated figures, tables, and the report are written to `results/`.

## Windows example

```powershell
& "D:\anaconda3\envs\cu130\python.exe" train_and_analyze.py `
  --epochs 8 `
  --batch-size 256 `
  --num-workers 4
```

If multiprocessing data loading causes an environment-specific error on Windows, set `--num-workers 0`.
