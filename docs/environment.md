# Local environment

Captured 2026-09-26, Asia/Kolkata. Commands were run in native Windows PowerShell, not WSL. This does not establish whether WSL is installed elsewhere on the machine.

Hardware: NVIDIA GeForce RTX 4050 Laptop GPU, 6141 MiB VRAM, driver 592.82. Keep a separate 4 GiB compatibility test; 6 GB available hardware does not prove that target passes.

## nvidia-smi output

```text
Sat Sep 26 02:01:03 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 592.82                 Driver Version: 592.82         CUDA Version: 13.1     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4050 ...  WDDM  |   00000000:01:00.0 Off |                  N/A |
| N/A   34C    P8              1W /   30W |       0MiB /   6141MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A           14860    C+G   ...2p2nqsd0c76g0\app\ChatGPT.exe      N/A      |
+-----------------------------------------------------------------------------------------+
```

The displayed CUDA 13.1 is the driver's supported CUDA level, not evidence that a CUDA toolkit or CUDA-enabled PyTorch is installed.

## python --version output

```text
Python 3.13.14
```

Resolved executable: C:\Users\HP\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\python.exe

## OS output

Command: Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,OSArchitecture | Format-List

```text

Caption        : Microsoft Windows 11 Home Single Language
Version        : 10.0.26200
OSArchitecture : 64-bit
```

Python platform probe: Windows-11-10.0.26200-SP0. RuntimeInformation reports Microsoft Windows 10.0.26200; that kernel version does not mean the product is Windows 10.

## Dependency probe

```text
torch: 2.13.0+cpu
CUDA build: None
CUDA available: False
module present: torch=True, numpy=True
module absent: rasterio, pystac_client, yaml
```

These probes describe the current interpreter only. They are not an installation or model benchmark. Python install-manager inventory also reported Python 3.14.2; the manager updated itself to 26.3 during that inventory command. No TrustSR project dependencies were installed.

## Reproducibility plan

Create an isolated Python 3.11 environment for the first compatibility trial (proposed, not yet installed/tested), without altering the existing interpreter. Resolve compatible CUDA PyTorch, SEN2SR, mlstac, safetensors, NumPy and rasterio/GDAL wheels; pin the successfully tested versions and record pip check/freeze. Upstream SEN2SR currently declares NumPy >=2.0.2; do not pair an old compiled dependency with it without testing.

On native Windows use Windows binary wheels for PyTorch and rasterio/GDAL and verify COG support. WSL, if deliberately adopted later, needs a separate Linux environment and dependency lock; do not mix Windows and Linux packages or copy their installation commands blindly. Colab is a separate Linux runtime and must record its actual GPU, CUDA and package versions on every run. The full Mamba-based SEN2SR installation is outside the chosen lite path.

After installation, require torch.cuda.is_available(), a real GPU tensor operation, rasterio import/COG write-read, then R1. Never report CPU inference or this nvidia-smi output as successful GPU model testing.

The workspace was not a Git repository at initial inspection. No commit was possible at that point. Project source/data paths are on OneDrive; keep large downloads/checkpoints out of Git and account for local disk and sync costs before fetching archives.
