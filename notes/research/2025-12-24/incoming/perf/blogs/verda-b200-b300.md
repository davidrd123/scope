Here are the technical notes derived from the Verda AI blog post regarding the NVIDIA Blackwell B200 and B300 architecture.

***

# NVIDIA B200 & B300: GPU Architecture and Software Stack
**Source:** Verda AI (formerly DataCrunch)
**Date:** Dec 17, 2025
**Topic:** Blackwell Architecture Specification & Setup

## 1. B200 vs. B300 (Ultra) Comparison
The Verda VM line introduces the Blackwell family. The **B300** is the high-performance "Ultra" variant.

| Feature | B200 | B300 (Ultra) | Delta (B300 vs B200) |
| :--- | :--- | :--- | :--- |
| **FP4 Dense Perf** | 9.0 PetaFLOPS | **14 PetaFLOPS** | +55.6% Faster |
| **FP64 Perf** | **37 TeraFLOPS** | 1.25 TeraFLOPS | **B300 has negligible FP64** |
| **Memory Capacity** | 180 GB HBM3E | **288 GB HBM3E** | +55.6% Larger |
| **Bandwidth** | 7.7 TB/s | **8 TB/s** | Slightly Faster |
| **TDP (Power)** | 1000W | 1100W | Requires better cooling |

**Key Takeaways:**
*   **B300** is optimized for AI inference/training (FP4/FP8) with massive memory but sacrifices FP64 (HPC) performance.
*   **Form Factor:** References "SXM6 B300" on an HGX board, distinct from the on-chip GB300s found in NVL72 racks.

## 2. Blackwell Architecture Features
New hardware capabilities introduced in the Blackwell generation (Compute Capability 10.x / SM103).

### A. 5th-Generation Tensor Cores
*   **Precision:** Supports FP8, FP6, and **FP4**.
*   **Speed:** Cores are larger and ~2–2.5x faster than H100.
*   **Utilization:** Relies heavily on TMA (Tensor Memory Accelerator) to keep cores fed; loading data efficiently is even more critical than on Hopper.

### B. Tensor Memory (TMEM)
A new memory space introduced in Blackwell, residing conceptually between Registers and Shared Memory.
*   **Function:** Very similar to register memory but requires explicit user management/allocation.
*   **Usage:** Reserved for advanced kernel programming (e.g., ThunderKittens).
*   **Goal:** Optimizes data flow without changing the logical algorithm of the kernel.

### C. CTA Pairs (2CTA) & Cluster Cooperation
*   **Concept:** The PTX model allows **two Cooperative Thread Arrays (CTAs)** to execute tensor core instructions together.
*   **Capability:** One CTA can access the **TMEM** of the other CTA in the pair.
*   **Benefit:** Higher efficiency for matrix operations.

### D. Memory Hierarchy
The hierarchy from fastest/smallest to slowest/largest:
1.  **Registers (RMEM):** Fastest, spills if full.
2.  **Tensor Memory (TMEM):** *New.* Explicitly managed, Blackwell specific.
3.  **Shared Memory (SMEM):** Fast, partitioned per Block/Cluster.
4.  **L2 Cache:** ~100MBs, shared between SMs.
5.  **Global Memory (GMEM):** HBM3E, huge capacity, slowest latency.

## 3. Software Stack & Environment
Verda VMs provide a pre-configured environment for Blackwell.

*   **OS:** Ubuntu 24.04
*   **CUDA Toolkit:** 13.0 or 12.8
*   **Drivers:** Version 580+
*   **Containerization:** Docker & NVIDIA Container Toolkit pre-installed.

**Docker Run Command:**
```bash
docker run --rm --gpus all --pull=always ubuntu nvidia-smi
```

**NCCL Testing:**
To verify system interconnects:
```bash
sudo apt install -y openmpi-bin libopenmpi-dev
# Clone and build NCCL tests
git clone https://github.com/NVIDIA/nccl-tests.git /home/ubuntu/nccl-tests
cd /home/ubuntu/nccl-tests
make MPI=1 MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi -j$(nproc)
# Run test
OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 mpirun -np 4 ...
```

## 4. Profiling Configuration
Accessing hardware counters on cloud GPUs often requires root, but Verda configures VMs to allow user-level profiling.

**Tools Supported:** `ncu` (Nsight Compute), Nsight Systems, Proton (Triton profiler).

**Config File:** `/etc/modprobe.d/nvidia-profiling.conf`
**Setting:**
```bash
options nvidia NVreg_RestrictProfilingToAdminUsers=0
```

## 5. Troubleshooting (Annex I)
**Issue:** PyTorch/Triton compilation errors on B300 (SM103).
*   **Error:** `ptxas fatal : Value 'sm_103a' is not defined`
*   **Cause:** Stable PyTorch versions (e.g., 2.9.1) bundle a Triton version where `ptxas` is too old to recognize the Blackwell architecture.

**Solution 1: Install Nightly**
PyTorch Nightly (post-Dec 2025) usually fixes this.
```bash
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu130
```

**Solution 2: Patching `ptxas` Path**
If building from source or using specific versions, point Triton to the system `ptxas` (which should be CUDA 13.0+) rather than the bundled one.

*Script to create a compatible venv:*
```bash
#!/bin/bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sudo env UV_INSTALL_DIR=/usr/local/bin sh

# Create venv
uv venv torch --python 3.12

# Patch activation script to export TRITON_PTXAS_PATH
echo "export TRITON_PTXAS_PATH=\"\$(which ptxas)\"" >> ~/torch/bin/activate

# Activate and Install
source ~/torch/bin/activate
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```
