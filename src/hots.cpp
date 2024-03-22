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
#include <sycl/sycl.hpp>
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

using namespace std;
using namespace cl;

class hots_kernel;

sycl::event hots(sycl::queue &q, unsigned int *events_buf, float *modelCenters_buf, unsigned int *hist_buf){
    auto queue_event = q.submit([&](auto &cgh) {
    
            //Call the kernel
            cgh.template single_task<hots_kernel>([=]() [[intel::kernel_args_restrict]]{
                
                unsigned int acci = 0, art = 0;

                unsigned int i = 0, j = 0, k = 0, l = 0, rel_pos = 0, tcbp = 0, tci = 0, diameter = 0, nFeatures =  0, radius, tau, local_hist[F_FEAT];
                unsigned int j_1 = 0, k_1 = 0, l_1 = 0, rel_pos_1 = 0, tci_1 = 0, diameter_1 = 0;
                unsigned int j_2 = 0, k_2 = 0, l_2 = 0, rel_pos_2 = 0, tci_2 = 0, diameter_2 = 0;
                
                int nbhXS = 0, nbhYS = 0, nbhXE = 0, nbhYE = 0;
                int nbhXS_1 = 0, nbhYS_1 = 0, nbhXE_1 = 0, nbhYE_1 = 0;
                int nbhXS_2 = 0, nbhYS_2 = 0, nbhXE_2 = 0, nbhYE_2 = 0;
                
                unsigned int model_c_offset = 0;

                float timeSurface = 0.0, res = 0.0, distance = 10000000.0, ts_inter, res_inter, modelCenters_local[NF * DIMS0];
                float timeSurface_1 = 0.0, res_1 = 0.0, distance_1 = 10000000.0, ts_inter_1, res_inter_1,  modelCenters_local_1[NF * KN * DIMS1];
                float timeSurface_2 = 0.0, res_2 = 0.0, distance_2 = 10000000.0, ts_inter_2, res_inter_2, modelCenters_local_2[NF * KN * 2 * DIMS2];

                struct nEvent_fpga cur_event = {0} ;
                struct nEvent_fpga cur_event_1 = {0};
                struct nEvent_fpga cur_event_2 = {0};

                struct nEvent_fpga pxState[TCB] = {{0}};
                struct nEvent_fpga pxState_1[TCB] = {{0}};
                struct nEvent_fpga pxState_2[TCB] = {{0}};
                
                float timeSurfaceVec[DIMS0];
                float timeSurfaceVec_1[DIMS1];
                float timeSurfaceVec_2[DIMS2];
                
                // copy data from RAM to local memory
                #pragma speculated_iterations 1936
                for(acci = 0; acci < NF * DIMS0; acci++)
                {
                    modelCenters_local[acci] = modelCenters_buf[acci];
                }
                
                #pragma speculated_iterations 14112
                for(acci = 0; acci < NF * KN * DIMS1; acci++)
                {
                    modelCenters_local_1[acci] = modelCenters_buf[acci + (NF * DIMS0)];
                }

                #pragma speculated_iterations 34848
                for(acci = 0; acci < NF * KN * 2 * DIMS2; acci++)
                {
                    modelCenters_local_2[acci] = modelCenters_buf[acci + ((NF * KN * DIMS1) + (NF * DIMS0))];
                }
                
                cur_event = {0} ;
                cur_event_1 = {0};
                cur_event_2 = {0};
                    
                acci = 0;
                std::fill(local_hist, local_hist + (F_FEAT), 0);
               
                // main control loop for one sample
                #pragma speculated_iterations 100
                for(i = 0; i < EVPS; i++)
                {
                    // Layer one
                    acci = i;
                    tcbp = i % TCB;
                    
                    // pull data from stream into our local memory
                    cur_event.x = events_buf[(acci * 4)];
                    cur_event.y = events_buf[(acci * 4) + 1];
                    cur_event.p = events_buf[(acci * 4) + 2];
                    cur_event.t = events_buf[(acci * 4) + 3];

                    pxState[acci].x = cur_event.x;
                    pxState[acci].y = cur_event.y;
                    pxState[acci].p = cur_event.p;
                    pxState[acci].t = cur_event.t;

                    // time surface bounds
                    nbhXS = cur_event.x - RAD;	// left most neighborhood bound
                    nbhYS = cur_event.y - RAD; 	// top most neighborhood bound
                    nbhXE = cur_event.x + RAD;	// right most neighborhood bound
                    nbhYE = cur_event.y + RAD;	// bottom most neighborhood bound

                    // in sensor pixel bounds checks
                    if(nbhXS < 0){nbhXS = 0;}
                    if(nbhYS < 0){nbhYS = 0;}
                    if(nbhXE >= WIDTH){nbhXE = WIDTH - 1;}
                    if(nbhYE >= HEIGHT){nbhYE = HEIGHT - 1;}
                    
               
                    // check neighborhood for time-context
                    diameter = (RAD * 2) + 1;
                    std::fill(timeSurfaceVec, timeSurfaceVec + (DIMS0), 0.0);
                    #pragma speculated_iterations 100
                    for(j = 0; j < TCB; j++)
                    {                
                        if(j >= i)// TCB used for circular buffer implementation
                        {
                            continue;
                        }
                        
                        tci = j;
                        //if(i < TCB){tci = j;}else{tci = (tcbp + j + 1) % TCB;}

                        // populate time surface with buffer events
                        if(pxState[tci].p == cur_event.p && (pxState[tci].y >= nbhYS && pxState[tci].y <= nbhYE) && (pxState[tci].x >= nbhXS && pxState[tci].x <= nbhXE))
                        {
                            if(cur_event.t > pxState[tci].t){
                                ts_inter = -((float)cur_event.t - (float)pxState[tci].t)/(float)(TAU * 1000);
                            }
                            else{
                                ts_inter = -((float)pxState[tci].t - (float)cur_event.t)/(float)(TAU * 1000); //should never happen
                            }
                   
                            timeSurface = cl::sycl::exp(ts_inter);            
                            timeSurfaceVec[(((diameter * diameter) - 1) / 2) + (pxState[tci].x - cur_event.x) + ((pxState[tci].y - cur_event.y)*diameter)] = timeSurface;
                        }
                    }
                    
                    // fill remaining time surface points
                    for(j = nbhYS; j < nbhYE; j++)
                    {
                        for(k = nbhXS; k < nbhXE; k++)
                        {
                            rel_pos = (k + (diameter * j));
                            
                            if(timeSurfaceVec[rel_pos] == 0 )
                            {
                                ts_inter = -((float)cur_event.t)/(float)(TAU * 1000);

                                timeSurface = cl::sycl::exp(ts_inter);
                                timeSurfaceVec[rel_pos] = timeSurface;
                            }
                        }
                    }
                              
                    // calculate new polarity
                    distance = std::numeric_limits<float>::max();

                    #pragma speculated_iterations 16
                    for(j = 0; j < NF; j++)
                    {
                        // calculate distance from center
                        res = 0.0;

                        #pragma speculated_iterations 121
                        for(k = 0; k < DIMS0; k++)
                        {
                            res_inter = timeSurfaceVec[k] - modelCenters_local[(j * DIMS0) + k];
                            res += cl::sycl::pow(res_inter, (float) 2.0);
                        }
                        res = cl::sycl::sqrt(res);

                        if(res < distance)
                        {
                            cur_event.p = j;
                            distance = res;
                        }
                    }

                    // Layer Two
                    cur_event_1.x = cur_event.x;
                    cur_event_1.y = cur_event.y;
                    cur_event_1.p = cur_event.p;
                    cur_event_1.t = cur_event.t;

                    pxState_1[acci].x = cur_event_1.x;
                    pxState_1[acci].y = cur_event_1.y;
                    pxState_1[acci].p = cur_event_1.p;
                    pxState_1[acci].t = cur_event_1.t;


                    // time surface neighborhood bounds
                    nbhXS_1 = cur_event_1.x - (RAD * KR);	// left most neighborhood bound
                    nbhYS_1 = cur_event_1.y - (RAD * KR); 	// top most neighborhood bound
                    nbhXE_1 = cur_event_1.x + (RAD * KR);	// right most neighborhood bound
                    nbhYE_1 = cur_event_1.y + (RAD * KR);	// bottom most neighborhood bound

                    // in sensor pixel bounds checks
                    if(nbhXS_1 < 0){nbhXS_1 = 0;}
                    if(nbhYS_1 < 0){nbhYS_1 = 0;}
                    if(nbhXE_1 >= WIDTH){nbhXE_1 = WIDTH - 1;}
                    if(nbhYE_1 >= HEIGHT){nbhYE_1 = HEIGHT - 1;}

                    // calculate time context
                    diameter_1 = (RAD * KR * 2) + 1;
                    std::fill(timeSurfaceVec_1, timeSurfaceVec_1 + (DIMS1), 0.0);
                    #pragma speculated_iterations 100
                    for(j_1 = 0; j_1 < TCB; j_1++)
                    {
                        if(j_1 >= i)// TCB used for circular buffer implementation
                        {
                            continue;
                        }
                        
                        tci_1 = j_1;
                        //if(i < TCB){tci_1 = j_1;}else{tci_1 = (tcbp + j_1 + 1) % TCB;}
    
                        // populate time surface with buffer events
                        if(pxState_1[tci_1].p == cur_event_1.p && (pxState_1[tci_1].y >= nbhYS_1 && pxState_1[tci_1].y <= nbhYE_1) && (pxState_1[tci_1].x >= nbhXS_1 && pxState_1[tci_1].x <= nbhXE_1))
                        {
                            if(cur_event_1.t > pxState_1[tci_1].t){
                                ts_inter_1 = -((float)cur_event_1.t - (float)pxState_1[tci_1].t)/(float)(TAU * KT * 1000);
                            }
                            else{
                                ts_inter_1 = -((float)pxState_1[tci_1].t - (float)cur_event_1.t)/(float)(TAU * KT * 1000); //should never happen
                            }
                           
                            timeSurface_1 = cl::sycl::exp(ts_inter_1);
                            timeSurfaceVec_1[(((diameter_1 * diameter_1) - 1) / 2) + (pxState_1[tci_1].x - cur_event_1.x) + ((pxState_1[tci_1].y - cur_event_1.y)*diameter_1)] = timeSurface_1;
                        }
                    }
                    
                    // fill remaining time surface points
                   for(j_1 = nbhYS_1; j_1 < nbhYE_1; j_1++)
                   {
                        for(k_1 = nbhXS_1; k_1 < nbhXE_1; k_1++)
                        {
                            rel_pos_1 = (k_1 + (diameter_1 * j_1));
                            
                            if(timeSurfaceVec_1[rel_pos_1] == 0)
                            {
                                ts_inter_1 = -((float)cur_event_1.t)/(float)(TAU * KT * 1000);

                                timeSurface_1 = cl::sycl::exp(ts_inter_1);
                                timeSurfaceVec_1[rel_pos_1] = timeSurface_1;
                            }
                        }
                            
                    }

                    // calculate new polarity
                    distance_1 = std::numeric_limits<float>::max();

                    #pragma speculated_iterations 32
                    for(j_1 = 0; j_1 < NF * KN; j_1++)
                    {
                        // calculate distance from center
                        res_1 = 0.0;

                        #pragma speculated_iterations 441
                        for(k_1 = 0; k_1 < DIMS1; k_1++)
                        {
                            res_inter_1 = timeSurfaceVec_1[k_1] - modelCenters_local_1[(j_1 * DIMS1) + k_1];
                            res_1 += cl::sycl::pow(res_inter_1, (float) 2.0);
                        }
                        res_1 = cl::sycl::sqrt(res_1);
                        
                        if(res_1 < distance_1)
                        {
                            cur_event_1.p = j_1;
                            distance_1 = res_1;
                        }
                    }

                    // Layer Three
                    cur_event_2.x = cur_event_1.x;
                    cur_event_2.y = cur_event_1.y;
                    cur_event_2.p = cur_event_1.p;
                    cur_event_2.t = cur_event_1.t;

                    pxState_2[acci].x = cur_event_2.x;
                    pxState_2[acci].y = cur_event_2.y;
                    pxState_2[acci].p = cur_event_2.p;
                    pxState_2[acci].t = cur_event_2.t;


                    // time surface neighborhood bounds
                    nbhXS_2 = cur_event_2.x - (RAD * KR * 2);	// left most neighborhood bound
                    nbhYS_2 = cur_event_2.y - (RAD * KR * 2); 	// top most neighborhood bound
                    nbhXE_2 = cur_event_2.x + (RAD * KR * 2);	// right most neighborhood bound
                    nbhYE_2 = cur_event_2.y + (RAD * KR * 2);	// bottom most neighborhood bound

                    // in sensor pixel bounds checks
                    if(nbhXS_2 < 0){nbhXS_2 = 0;}
                    if(nbhYS_2 < 0){nbhYS_2 = 0;}
                    if(nbhXE_2 >= WIDTH){nbhXE_2 = WIDTH - 1;}
                    if(nbhYE_2 >= HEIGHT){nbhYE_2 = HEIGHT - 1;}

                    
                     // calculate time context
                    diameter_2 = (RAD * KR * 2 * 2) + 1;
                    std::fill(timeSurfaceVec_2, timeSurfaceVec_2 + (DIMS2), 0.0);
                    #pragma speculated_iterations 100
                    for(j_2 = 0; j_2 < TCB; j_2++)
                    {
                        if(j_2 >= i)// TCB used for circular buffer implementation
                        {
                            continue;
                        }
                        
                        tci_2 = j_2;
                        //if(i < TCB){tci_2 = j_2;}else{tci_2 = (tcbp + j_2 + 1) % TCB;}

                         // populate time surface with buffer events
                        if(pxState_2[tci_2].p == cur_event_2.p && (pxState_2[tci_2].y >= nbhYS_2 && pxState_2[tci_2].y <= nbhYE_2) && (pxState_2[tci_2].x >= nbhXS_2 && pxState_2[tci_2].x <= nbhXE_2))
                        {
                            if(cur_event_2.t > pxState_2[tci_2].t){
                                ts_inter_2 = -((float)cur_event_2.t - (float)pxState_2[tci_2].t)/(float)(TAU * KT * 2000);
                            }
                            else{
                                ts_inter_2 = -((float)pxState_2[tci_2].t - (float)cur_event_2.t)/(float)(TAU * KT * 2000); //should never happen
                            }

                            timeSurface_2 = cl::sycl::exp(ts_inter_2);
                            timeSurfaceVec_2[(((diameter_2 * diameter_2) - 1) / 2) + (pxState_2[tci_2].x - cur_event_2.x) + ((pxState_2[tci_2].y - cur_event_2.y)*diameter_2)] = timeSurface_2;
                        }
                    }
                    
                    // fill remaining time surface points
                    for(j_2 = nbhYS_2; j_2 < nbhYE_2; j_2++)
                    {
                        for(k_2 = nbhXS_2; k_2 < nbhXE_2; k_2++)
                        {
                            rel_pos_2 = (k_2 + (diameter_2 * j_2));
                            
                            if(timeSurfaceVec_2[rel_pos_2] == 0)
                            {
                                ts_inter_2 = -((float)cur_event_2.t)/(float)(TAU * KT * 2000);

                                timeSurface_2 = cl::sycl::exp(ts_inter_2);
                                timeSurfaceVec_2[rel_pos_2] = timeSurface_2;
                            }
                        }
                            
                    }
                    
                    // calculate new polarity
                    distance_2 = std::numeric_limits<float>::max();
                     
                    #pragma speculated_iterations 64
                    for(j_2 = 0; j_2 < NF * KN * 2; j_2++)
                    {
                        // calculate distance from center
                        res_2 = 0.0;
                          
                        #pragma speculated_iterations 1681
                        for(k_2 = 0; k_2 < DIMS2; k_2++)
                        {
                            res_inter_2 = timeSurfaceVec_2[k_2];
                            res_inter_2 -= modelCenters_local_2[(j_2 * DIMS2) + k_2];
                            res_2 += cl::sycl::pow(res_inter_2, (float) 2.0);
                        }
                        res_2 = cl::sycl::sqrt(res_2);
                            
                        if(res_2 < distance_2)
                        {
                            cur_event_2.p = j_2;
                            distance_2 = res_2;
                        }
                    }
                    
                    local_hist[cur_event_2.p] = local_hist[cur_event_2.p] + 1;
                }
                

                // write back to ram buffer
                #pragma unroll
                #pragma speculated_iterations 64
                for(acci = 0; acci < F_FEAT; acci++)
                {
                    hist_buf[acci] = local_hist[acci];
                }

            });

        });
    
    return queue_event;
}
    

int main() {

    //Init
    unsigned int i =  0, j = 0, k = 0, l = 0, total_centers = ((DIMS*NF)+(DIMS1*NF*KN)+(DIMS2*NF*KN*2)), eventsInStream = 0, prev_time = 0, n_feat = 0, model_c_offset = 0, hDims = 0;
    struct nEvent_t cur_event;

    //Define input/ouput arrays
    unsigned int ref_output[F_FEAT];
    unsigned int *events_data = static_cast<unsigned int *>(malloc(sizeof(unsigned int) * 4 * EVPS));
    unsigned int *hist_data = static_cast<unsigned int *>(malloc(sizeof(unsigned int) * F_FEAT));
    float *modelCenters_data= static_cast<float *>(malloc(sizeof(float) * total_centers));
    std::fill(modelCenters_data, modelCenters_data + (total_centers), 0.0);

    //Initialize the output array
    std::fill(hist_data, hist_data + (F_FEAT), 0);
    
    // parse model file for buffer
    readModel();
 
    // fill flat buffer
    model_c_offset = 0;
    for(j = 0; j < LAYERS; j++)
    {
        if(j == 0)
        {
            n_feat = NF;
        }
        else
        {
            n_feat = NF * KN * j;
        }
        
        if(j == 0)
        {
            hDims = DIMS0;
        }
        else if(j == 1)
        {
            hDims = DIMS1;
        }
        else
        {
            hDims = DIMS2;
        }

        if(j == 1)
        {
            model_c_offset += (NF * DIMS0);
        }
        else if(j > 1)
        {
            model_c_offset += (NF * KN * (j - 1) * DIMS1);
        }

        for(k = 0; k < n_feat; k++)
        {
            
            for(l = 0; l < hDims; l++)
            {
                modelCenters_data[model_c_offset + (k * hDims) + l] = modelCenters[j][k][l];
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
    string test = "data/03539.bin";
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
        sycl::queue device_queue(device_selector, NULL, property_list);

        //Buffer setup
        unsigned int *events_buf = malloc_device<unsigned int>(4 * EVPS, device_queue);
        float *modelCenters_buf = malloc_device<float>(total_centers, device_queue);
        unsigned int *hist_buf = malloc_device<unsigned int>(F_FEAT, device_queue);

        // copy data to buffers
        device_queue.memcpy(events_buf, events_data, sizeof(unsigned int) * 4 * EVPS).wait();
        device_queue.memcpy(modelCenters_buf, modelCenters_data, sizeof(float) * total_centers).wait();

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
        
        // copy data back to host
        device_queue.memcpy(hist_data, hist_buf, sizeof(unsigned int) * F_FEAT).wait();
        std::cout << "Waiting for results to transfer..." << std::endl;
        
        t1_kernel = q_event.get_profiling_info<sycl::info::event_profiling::command_start>();
        t2_kernel = q_event.get_profiling_info<sycl::info::event_profiling::command_end>();
        std::cout << "Kernel Start: " << t1_kernel << " : End: " << t2_kernel <<std::endl;
        time_kernel = (t2_kernel - t1_kernel) / NS;
        std::cout << "Kernel execution time: " << time_kernel << " seconds" << std::endl;
        
        sycl::free(events_buf, device_queue);
        sycl::free(modelCenters_buf, device_queue);
        sycl::free(hist_buf, device_queue);
    }
    
    //Test the results against the golden results -- only useful for testing one file (03539.bin)
    std::fill(ref_output, ref_output + (F_FEAT), 0);
    ref_output[0] = 0;
    ref_output[1] = 2;
    ref_output[2] = 1;
    ref_output[3] = 0;
    ref_output[4] = 0;
    ref_output[5] = 4;
    ref_output[6] = 0;
    ref_output[7] = 6;
    ref_output[8] = 0;
    ref_output[9] = 0;
    ref_output[10] = 0;
    ref_output[11] = 15;
    ref_output[12] = 0;
    ref_output[13] = 0;
    ref_output[14] = 0;
    ref_output[15] = 0;
    ref_output[16] = 0;
    ref_output[17] = 0;
    ref_output[18] = 0;
    ref_output[19] = 1;
    ref_output[20] = 0;
    ref_output[21] = 0;
    ref_output[22] = 0;
    ref_output[23] = 0;
    ref_output[24] = 0;
    ref_output[25] = 0;
    ref_output[26] = 0;
    ref_output[27] = 3;
    ref_output[28] = 0;
    ref_output[29] = 1;
    ref_output[30] = 1;
    ref_output[31] = 0;
    ref_output[32] = 0;
    ref_output[33] = 0;
    ref_output[34] = 0;
    ref_output[35] = 2;
    ref_output[36] = 0;
    ref_output[37] = 0;
    ref_output[38] = 0;
    ref_output[39] = 0;
    ref_output[40] = 0;
    ref_output[41] = 0;
    ref_output[42] = 4;
    ref_output[43] = 0;
    ref_output[44] = 0;
    ref_output[45] = 0;
    ref_output[46] = 1;
    ref_output[47] = 0;
    ref_output[48] = 0;
    ref_output[49] = 5;
    ref_output[50] = 0;
    ref_output[51] = 5;
    ref_output[52] = 27;
    ref_output[53] = 4;
    ref_output[54] = 3;
    ref_output[55] = 0;
    ref_output[56] = 0;
    ref_output[57] = 0;
    ref_output[58] = 6;
    ref_output[59] = 0;
    ref_output[60] = 0;
    ref_output[61] = 9;
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
    int i = 0, j = 0, k = 0, nFeatures = 0, hDims = 0;
    bool warn = true;
    vector<float> tempPoint {};
    vector<vector<float>> layerCenter {};
    modelFile.open("models/custom-100-5R.csv");

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
    for(i = 0; i < LAYERS; i++)
    {
    // layer specific parameters
    if(i == 0)
    {
              nFeatures = NF;
    }
    else
      {
            nFeatures = NF * KN * i;
    }
    
    if(i == 0)
    {
        hDims = DIMS0;
    }
    else if(i == 1)
    {
        hDims = DIMS1;
    }
    else
    {
        hDims = DIMS2;
    }

    layerCenter = {};
    for(j = 0; j < nFeatures; j++)
    {
      tempPoint = {};
      for(k = 0; k < hDims; k++)
      {
        // last line is delimted by \n not ,
        if(k < hDims - 1)
        {
          getline(modelFile,temp, ',');
        }
        else
        {
          getline(modelFile,temp, '\n');
        }
        tempPoint.push_back(stof(temp));
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
        //ann = cv::ml::ANN_MLP::load("hots/models/perc/ann100-1000.model");
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
