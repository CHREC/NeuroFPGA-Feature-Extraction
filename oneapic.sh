#!/bin/bash
export PATH=/glob/intel-python/python3/bin/:/glob/intel-python/python2/bin/:${PATH}
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
#source /glob/development-tools/versions/oneapi/2022.2/oneapi/setvars.sh --force
cd $HOME/oneAPI/research/
icpx -std=c++20 -fsycl -I $INTELFPGAOCLSDKROOT/include/ref -qactypes -fintelfpga -no-fma -fp-model=precise -Xshardware -Xsv -Xstarget=/opt/intel/oneapi/intel_s10sx_pac:pac_s10 src/hots.cpp src/NeuroEventReader.cpp -Xsprofile -v -o fpga-hots-100-5-64-f.hw 
#icpx -fsycl -I $INTELFPGAOCLSDKROOT/include/ref -qactypes -fintelfpga -no-fma -fp-model=precise -Xshardware -Xsv -Xstarget=$HOME/oneAPI/research/ia840f_ofshldshim/ia840f:ofs_ia840fr0 hots.cpp NeuroEventReader.cpp -Xsprofile -v -o fpga-hots-100-ax.hw 
# -no-fma -fp-model=precise
#qsub -l walltime=24:00:00 -l nodes=1:fpga_compile:ppn=2 -d . oneapic.sh
#qsub -I -l nodes=1:stratix10:ppn=2 -d . 
#aocl initialize acl0 pac_s10
#emu icpx -std=c++20 -fsycl -I $INTELFPGAOCLSDKROOT/include/ref -fintelfpga -DFPGA_EMULATOR -qactypes src/hots.cpp src/NeuroEventReader.cpp -o fpga.emu
#cpu icpx -std=c++20 -fsycl -I $INTELFPGAOCLSDKROOT/include/ref -DCPU_HOST -qactypes src/hots.cpp src/NeuroEventReader.cpp -o cpu.test
#rebuild host dpcpp -I $INTELFPGAOCLSDKROOT/include/ref -qactypes -fintelfpga -fsycl-link=early -no-fma -fp-model=precise -Xshardware -Xsboard=intel_s10sx_pac:pac_s10 hots.cpp NeuroEventReader.cpp -Xsclock=400MHz -Xsprofile -v -o fpga-nv-100.hw -reuse-exe=fpga-nv-100.hw
# ia840f