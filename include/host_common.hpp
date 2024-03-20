#ifndef HOST_COMMON_H
#define HOST_COMMON_H

#include <string>
#include <iostream>
#include <fstream>
#include <sycl/ext/intel/ac_types/ac_int.hpp>

using namespace std;

// struct containing neuromorphic event
struct nEvent_t
{
	unsigned int x;	// x position
	unsigned int y;	// y position
	unsigned int p;	// polarity
	unsigned int t;	// timestamp
};

// struct containing neuromorphic event
struct nEvent_fpga
{
	unsigned int x;	// x position
	unsigned int y;	// y position
	unsigned int p;	// polarity
	unsigned int t;	// timestamp
};

#endif /*HOST_COMMON_H*/
