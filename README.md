# VkdispatchPaperFigures

This repository contains the raw data and python code used to generate the figures for the vkdispatch paper.

The data is stored in the `test_data_nvidia/` and `test_data_macos/` folders, for the NVIDIA and macOS tests respectively. The code to generate the figures is in the `src/` folder.

## Testing Environment

### NVIDIA

The machine used for the NVIDIA tests was a server running Ubuntu 22.04 LTS. The hardware specifications of the machine are as follows:

| Component | Specification |
|-----------|---------------|
| CPU | AMD Ryzen Threadripper PRO 5975WX (32 cores) |
| RAM | 512GB |
| GPU | NVIDIA RTX 6000 Ada Generation (48GB VRAM) |

And the software versions used for testing were:

| Component | Version |
|-----------|---------------|
| OS | Ubuntu 22.04 LTS (Kernel: 5.15.0-113-generic) |
| GPU Driver | 550.90.07 |
| CUDA Toolkit | 12.0 |
| Vulkan Instance | 1.3.280 |
| Vulkan Driver | 1.3.277 |
| OpenCL | OpenCL 3.0 CUDA 12.4.131 |
| Python | 3.10.12 |
| vkdispatch | 0.1.0 |
| cuda-python | 13.2.0 |
| pyopencl | 2026.1.2 |

### macOS

The macOS tests were run on a MacBook Pro (14-inch, 2023).

| Component | Specification |
|-----------|---------------|
| SoC | Apple M2 Pro (10-core CPU, 16-core GPU) |
| RAM | 32GB Unified Memory |

And the software versions used for testing were:

| Component | Version |
|-----------|---------------|
| OS | macOS 15.7.4 (24G517) |
| MoltenVK | 1.4.1 (*patched) |
| Vulkan Instance | 1.2.334 |
| Vulkan Driver | 1.2.334 |
| OpenCL | 1.2 |
| Python | 3.11.4 |
| vkdispatch | 0.1.0 |
| pyopencl | 2026.1.2 |