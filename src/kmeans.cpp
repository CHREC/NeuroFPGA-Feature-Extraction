/*
Copyright (c) 2024 NSF Center for Space, High-performance, and Resilient Computing (SHREC) University of Pittsburgh. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS AS IS AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#include <CL/sycl.hpp>

/*#define BETA09 20200827
#if __SYCL_COMPILER_VERSION <= BETA09
  #include <CL/sycl/intel/fpga_extensions.hpp>
  namespace INTEL = sycl::intel;  // Namespace alias for backward compatibility
#else
  #include <CL/sycl/INTEL/fpga_extensions.hpp>
#endif*/

#include <sycl/ext/intel/fpga_extensions.hpp>
#include <chrono>
#include "../include/common.hpp"
#include "../include/host_common.hpp"
#include "../include/hots.hpp"
#include "../include/NeuroEventReader.hpp"
#include <fcntl.h>
#include <math.h>
#include <memory.h>
#include <stdint.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/uio.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <malloc.h>
#include <algorithm>

#define NS (1000000000.0) // number of nanoseconds in a second
// k-means version of accelerator

using namespace std;
using namespace cl;

class hots_kernel;

sycl::event hots(sycl::queue &q, sycl::buffer<unsigned int, 1> events_buf, sycl::buffer<float, 1> modelCenters_buf, sycl::buffer<unsigned int, 1> hist_buf){
    auto queue_event = q.submit([&](sycl::handler &cgh) {
            //Create accessors
            auto _eStream = events_buf.get_access<sycl::access::mode::read>(cgh);
            auto _modelCenters = modelCenters_buf.get_access<sycl::access::mode::read>(cgh);
            auto _hist = hist_buf.get_access<sycl::access::mode::read_write>(cgh);

            //Call the kernel
            cgh.single_task<hots_kernel>([=]() {
                
                unsigned int acci = 0;
                unsigned int i = 0, j = 0, k = 0, l = 0, m = 0, tcbp = 0, tci = 0, nFeatures =  0, radius, tau, local_hist[F_FEAT];
                unsigned int timeContext = 0, model_c_offset = 0;

                float timeSurface = 0.0, res = 0.0, distance = 10000000.0, ts_inter, res_inter, modelCenters_local[F_FEAT * DIMS];

                struct nEvent_fpga cur_event;

                struct nEvent_fpga pxState[TCB];

                float tempPoint[DIMS];

                //#pragma unroll	
                #pragma speculated_iterations 2
                for(i = 0; i < DIMS; i++){
                    tempPoint[i] = 0.0;
                }

                //#pragma unroll
                #pragma speculated_iterations 64
                for(i = 0; i <  F_FEAT; i++)
                {
                    local_hist[i] = 0;
                }
                
                // copy data from RAM to local memory
                //#pragma unroll
                #pragma speculated_iterations 128
                for(acci = 0; acci < F_FEAT * DIMS; acci++)
                {
                    modelCenters_local[acci] = _modelCenters[acci];
                }

                cur_event.x = 0;
                cur_event.y = 0;
                cur_event.p = 0;
                cur_event.t = 0;

                #pragma speculated_iterations 100
                for(i = 0; i < EVPS; i++)
                {
                    // Layer one
                    acci = i;
                    tcbp = i % TCB;

                    cur_event.x = _eStream[(acci * 4)];
                    cur_event.y = _eStream[(acci * 4) + 1];
                    cur_event.p = _eStream[(acci * 4) + 2];
                    cur_event.t = _eStream[(acci * 4) + 3];

                    pxState[tcbp].x = cur_event.x;
                    pxState[tcbp].y = cur_event.y;
                    pxState[tcbp].p = cur_event.p;
                    pxState[tcbp].t = cur_event.t;

                    tempPoint[0] = cur_event.x;
                    tempPoint[1] = cur_event.y;

                    // calculate new polarity
                    distance = 10000000.0;

                    #pragma speculated_iterations 64
                    for(j = 0; j < F_FEAT; j++)
                    {
                        // calculate distance from center
                        res = 0.0;
                        
                        #pragma unroll
                        #pragma speculated_iterations 2
                        for(k = 0; k < DIMS; k++)
                        {
                            res_inter = tempPoint[k] - modelCenters_local[(j * DIMS) + k];
                            res += cl::sycl::pow(res_inter, (float) 2);
                        }
                        res = cl::sycl::sqrt(res);

                        if(res < distance)
                        {
                            cur_event.p = j;
                            distance = res;
                        }
                    }

                    local_hist[cur_event.p] = local_hist[cur_event.p] + 1;
                }

                #pragma unroll
                #pragma speculated_iterations 64
                for(acci = 0; acci < F_FEAT; acci++)
                {
                    _hist[acci] = local_hist[acci];
                }
            });

        });
    
    return queue_event;
}
    

int main() {

    //Init
    unsigned int i =  0, j = 0, k = 0, l = 0, total_centers = F_FEAT, eventsInStream = 0, prev_time = 0, n_feat = 0, model_c_offset = 0;
    struct nEvent_t cur_event;

    //Define input/ouput arrays
    unsigned int events_data[4 * EVPS], hist_data[F_FEAT], ref_output[F_FEAT];
    float modelCenters_data[total_centers * DIMS];

    //Initialize the output array
    std::fill(hist_data, hist_data + (F_FEAT), 0);
    
    // parse model file for buffer
    readModel();

    // fill flat buffer
    model_c_offset = 0;
    for(j = 0; j < 1; j++)
    {
        if(j == 0)
        {
            n_feat = F_FEAT;
        }
        for(k = 0; k < n_feat; k++)
        {
            for(l = 0; l < DIMS; l++)
            {
                modelCenters_data[model_c_offset + (k * DIMS) + l] = modelCenters[j][k][l];
            }
        }
    }
    
    /* Reading dir ---- switch kernel to double buffer variant
    // vector of histogram (ordered) labels
    vector<string> labels {};
    
    for (const auto & entry : filesystem::directory_iterator("path/to/labeledfolders"))
    {
        labels.push_back(getClass((char *)entry.path().u8string().c_str()));

        cout << "\t\tDir " << labels.back() << "..." << endl;
    
        streamsInClass = 0;
        // enum files in class directory
        for (const auto & sample : filesystem::directory_iterator(entry.path()))
        {
            // check to see if we have read enough streams
            if(streamsInClass >= SPC)
            {
                break;
            }
    */
    
    //Read test stream
    string test = "data/504.bin";
    NeuroEventReader nEReader = NeuroEventReader(test);

    // reset on new stream
    eventsInStream = 0;
    prev_time = 0;
    cur_event = nEReader.readEvent();
    array< array<unsigned int, 4>, EVPS> tempES = {};

    while(true){
        // read event

        cur_event = nEReader.readEvent();

        // check if read last event
        if(cur_event.y == (HEIGHT + 1) && cur_event.x == (WIDTH + 1))
        {
            break;
        }


        /*if(prev_time > cur_event.t)
        {
            continue;
        }*/

        prev_time = cur_event.t;

        // save event for reordering
        tempES[eventsInStream][0] = cur_event.x;
        tempES[eventsInStream][1] = cur_event.y;
        tempES[eventsInStream][2] = cur_event.p;
        tempES[eventsInStream][3] = cur_event.t;
        
        eventsInStream++;
    }

    // make sure this stream has enough data
    if(eventsInStream < EVPS - 1)
    {
        cout << "Error: Not enough events in stream..." << endl;
    }
    
    // reorder events - ensure no future events come before current event
    std::sort(tempES.begin(), tempES.end(), timeSort);

    for(i = 0; i < EVPS; i++)
    {
        events_data[i * 4 + 0] = tempES[i][0];
        events_data[i * 4 + 1] = tempES[i][1];
        events_data[i * 4 + 2] = tempES[i][2];
        events_data[i * 4 + 3] = tempES[i][3];
    }

    

    //Block off this code
    //Putting all SYCL work within here ensures it concludes before this block
    //  goes out of scope. Destruction of the buffers is blocking until the
    //  host retrieves data from the buffer.
    {
        //Profiling setup
        //Set things up for profiling at the host
        chrono::high_resolution_clock::time_point t1_host, t2_host;
        sycl::event q_event;
        sycl::opencl::cl_ulong t1_kernel, t2_kernel;
        double time_kernel;
        auto property_list = sycl::property_list{sycl::property::queue::enable_profiling()};

        //Buffer setup
        //Define the sizes of the buffers
        //The sycl buffer creation expects a type of sycl:: range for the size
        sycl::range<1> num_events{4 * EVPS};
        sycl::range<1> num_modelCenters{total_centers * DIMS};
        sycl::range<1> num_hist{F_FEAT};

        //Create the buffers which will pass data between the host and FPGA
        sycl::buffer<unsigned int, 1> events_buf(events_data, num_events);
        sycl::buffer<float, 1> modelCenters_buf(modelCenters_data,num_modelCenters);
        sycl::buffer<unsigned int, 1> hist_buf(hist_data,num_hist);

        //Device selection
        //We will explicitly compile for the FPGA_EMULATOR, CPU_HOST, or FPGA
        #if defined(FPGA_EMULATOR)
          auto device_selector = sycl::ext::intel::fpga_emulator_selector_v;
          std::cout << "FPGA Emulator Selected!" << std::endl;
        #elif defined(CPU_HOST)
          auto device_selector = sycl::cpu_selector_v;
          std::cout << "CPU Selected!" << std::endl;
        #else
          auto device_selector = sycl::ext::intel::fpga_selector_v;
          std::cout << "FPGA Selected!" << std::endl;
        #endif

        //Create queue
        sycl::queue device_queue(device_selector,NULL,property_list);

        //Query platform and device
        sycl::platform platform = device_queue.get_context().get_platform();
        sycl::device device = device_queue.get_device();
        std::cout << "Platform name: " <<  platform.get_info<sycl::info::platform::name>().c_str() << std::endl;
        std::cout << "Device name: " <<  device.get_info<sycl::info::device::name>().c_str() << std::endl;

        std::cout << "Executing HOTS..." << std::endl;
        
        q_event = hots(device_queue, events_buf, modelCenters_buf, hist_buf);
       
        //Wait for the kernel to get finished before reporting the profiling
        device_queue.wait();
        std::cout << "Waiting for events to finish..." << std::endl;
        
        t1_kernel = q_event.get_profiling_info<sycl::info::event_profiling::command_start>();
        t2_kernel = q_event.get_profiling_info<sycl::info::event_profiling::command_end>();
        std::cout << "Kernel Start: " << t1_kernel << " : End: " << t2_kernel <<std::endl;
        time_kernel = (t2_kernel - t1_kernel) / NS;
        std::cout << "Kernel execution time: " << time_kernel << " seconds" << std::endl;
    }

    //Test the results against the golden results -- only useful for testing one file (504.bin)
    std::fill(ref_output, ref_output + (F_FEAT), 0);
    ref_output[0] = 0;
    ref_output[1] = 0;
    ref_output[2] = 0;
    ref_output[3] = 2;
    ref_output[4] = 0;
    ref_output[5] = 0;
    ref_output[6] = 0;
    ref_output[7] = 5;
    ref_output[8] = 4;
    ref_output[9] = 0;
    ref_output[10] = 0;
    ref_output[11] = 1;
    ref_output[12] = 2;
    ref_output[13] = 1;
    ref_output[14] = 0;
    ref_output[15] = 2;
    ref_output[16] = 0;
    ref_output[17] = 0;
    ref_output[18] = 0;
    ref_output[19] = 0;
    ref_output[20] = 0;
    ref_output[21] = 0;
    ref_output[22] = 1;
    ref_output[23] = 0;
    ref_output[24] = 0;
    ref_output[25] = 3;
    ref_output[26] = 0;
    ref_output[27] = 6;
    ref_output[28] = 4;
    ref_output[29] = 0;
    ref_output[30] = 0;
    ref_output[31] = 11;
    ref_output[32] = 0;
    ref_output[33] = 0;
    ref_output[34] = 0;
    ref_output[35] = 6;
    ref_output[36] = 0;
    ref_output[37] = 0;
    ref_output[38] = 0;
    ref_output[39] = 0;
    ref_output[40] = 0;
    ref_output[41] = 0;
    ref_output[42] = 11;
    ref_output[43] = 0;
    ref_output[44] = 0;
    ref_output[45] = 0;
    ref_output[46] = 0;
    ref_output[47] = 0;
    ref_output[48] = 0;
    ref_output[49] = 0;
    ref_output[50] = 0;
    ref_output[51] = 12;
    ref_output[52] = 0;
    ref_output[53] = 0;
    ref_output[54] = 0;
    ref_output[55] = 14;
    ref_output[56] = 0;
    ref_output[57] = 10;
    ref_output[58] = 3;
    ref_output[59] = 0;
    ref_output[60] = 0;
    ref_output[61] = 2;
    ref_output[62] = 0;
    ref_output[63] = 0;

    bool match = true;
    for (unsigned j = 0 ; j < F_FEAT; j++){
        if (hist_data[j] != ref_output[j]){
          std::cout << "Error: Result mismatch" << std::endl;
          std::cout << "j = " << j << " CPU result = " << ref_output[j] << " FPGA result = " << hist_data[j] << std::endl;
          match = false;
        }
    }

    std::cout << "TEST " << (match ? "PASSED" : "FAILED") << std::endl; 

    return 0;
}

bool timeSort(const array<unsigned int, 4> t1, const array<unsigned int, 4> t2)
{
    return t1[3] < t2[3];
}

string getClass(char * path)
{
    string clsNumStr;

    // extract label from path first - N-MNIST, second - off-train, third - class
    clsNumStr = strtok(path,"/");
    clsNumStr = strtok(NULL,"/");
    clsNumStr = strtok(NULL,"/");

    //update for absolute path
    clsNumStr = strtok(NULL,"/");
    clsNumStr = strtok(NULL,"/");
    clsNumStr = strtok(NULL,"/");
    clsNumStr = strtok(NULL,"/");
    clsNumStr = strtok(NULL,"/");

    return clsNumStr;
}

int readModel()
{
    ifstream modelFile;
    string temp;
    int i = 0, j = 0, k = 0, nFeatures = 0;
    bool warn = true;
    vector<float> tempPoint {};
    vector<vector<float>> layerCenter {};
    modelFile.open("models/custom-100-k.csv");

    labels = {};

    //read first line, parameters already known from #define
    getline(modelFile,temp, '\n');

    // parse in histogram order
    for(i = 0; i < CLASSES; i++)
    {
    // last line is delimted by \n not ,
    if(i < CLASSES - 1)
    {
      getline(modelFile,temp, ',');
    }
    else
    {
      getline(modelFile,temp, '\n');
    }

    labels.push_back(temp);
    }

    // read in centers for each layer
    for(i = 0; i < 1; i++)
    {
    // layer specific parameters
    if(i == 0)
    {
              nFeatures = F_FEAT;
    }
    else
      {
            nFeatures = NF * KN * i;
    }

    layerCenter = {};
    for(j = 0; j < nFeatures; j++)
    {
      tempPoint = {};
      for(k = 0; k < DIMS; k++)
      {
        // last line is delimted by \n not ,
        if(k < DIMS - 1)
        {
          getline(modelFile,temp, ',');
        }
        else
        {
          getline(modelFile,temp, '\n');
        }
        tempPoint.push_back(stod(temp));
      }
      layerCenter.push_back(tempPoint);
    }
    modelCenters.push_back(layerCenter);
    }

    // read in histograms

    switch(C_ALGO)
    {
    case 0:
      for(i = 0; i < CLASSES; i++)
      {
        for(j = 0; j < NF * KN * (LAYERS - 1); j++)
        {
          // last line is delimted by \n not ,
          if(j < (NF * KN * (LAYERS - 1)) - 1)
          {
            getline(modelFile,temp, ',');
          }
          else
          {
            getline(modelFile,temp, '\n');
          }

          modelHistograms[i][j] = stoi(temp);
        }
      }

      break;
    case 1:
        // parse model file
        //ann = cv::ml::ANN_MLP::load("models/perc/ann100-1000.model");
        break;
    default:
        break;
    }

    while(getline(modelFile,temp, ','))
    {
        if(warn)
        {
            cout << "Warning! The following lines were not parsed:" << endl;
            warn = false;
        }
        cout << temp << endl;
    }

    modelFile.close();

    return 0;
}
