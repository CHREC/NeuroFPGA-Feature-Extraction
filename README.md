# FPGA Acceleration of Feature-Extraction for Neuromorphic data

This repository contains FPGA accelerators for the spaito-temporal [HOTS: A Hierarchy of Event-Based Time-Surfaces for Pattern Recognition](https://ieeexplore.ieee.org/document/7508476) and spatial-only _k_-means feature-extraction algorithms. Designed using Intel's FPGA framework for oneAPI and SYCL. Implemented on Intel Programmable Acceleration Cards - Stratix 10 FPGAs (D5005).

More details about the accelerator architectures can be found at:
- [Clustering Classification on FPGAs for Neuromorphic Feature Extraction](https://ieeexplore.ieee.org/document/10171560)
- [Exploring ML-Oriented Hardware for Accelerated and Scalable Feature Extraction](https://d-scholarship.pitt.edu/45421/)

## Usage

The included files are intended for use with [Intel's DevCloud Platform for oneAPI](https://devcloud.intel.com/oneapi/get_started/). Designs have been successfully synthesized and tested on Stratix 10 FPGA nodes using Intel(R) oneAPI DPC++/C++ Compiler 2024.0.1.

### Testing and Compilation

The `oneapi-*-c.sh` scripts contain instructions on how to CPU simulate, FPGA emulate, and FPGA compile for both algorithms, where * is either `hots` or `kmeans`. Pretrained models are included for both algorithms, as well as testing samples for validation. Datasets should be placed in the `data` subdirectory and the double-buffering variants should be used for optimized streaming performance. 

#### Common Variable Controls

Both algorithms are by default configured using 100 events per object and 64 final features -- included models' training parameters. The included `HOTS` model is expecting N-MNIST data (data source = 0) and the inlcuded `kmeans` model is expecting VGA data (data source = 1). 

### Known Limitations

The CPU simulation and FPGA emulation for HOTS will occasionally result in segmentation faults or incorrect output. Most of the time, the simulation/emulation should pass.

## Citing

Please use the following citation when referencing this material:
>L. Kljucaric and A. D. George, "Clustering Classification on FPGAs for Neuromorphic Feature Extraction," 2023 IEEE 31st Annual International Symposium on Field-Programmable Custom Computing Machines (FCCM), Marina Del Rey, CA, USA, 2023, doi: 10.1109/FCCM57271.2023.00051

## Acknowledgements

This research was supported by SHREC industry and agency members and by the IUCRC Program of the National Science Foundation under Grant No. CNS-1738783. (University of Pittsburgh and Intel Corp.)

## Copyright

Copyright (c) 2024 NSF Center for Space, High-performance, and Resilient Computing (SHREC) University of Pittsburgh. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS AS IS AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.