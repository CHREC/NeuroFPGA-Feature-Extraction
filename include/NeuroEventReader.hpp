#ifndef NEUROEVENTREADER_H
#define NEUROEVENTREADER_H

#include "common.hpp"
#include "host_common.hpp"

class NeuroEventReader
{
	private:
		//ifstream tmpFile;
	    ifstream *evtData;// = ifstream();
	    int events;
	    unsigned int prev;

	public:
	    NeuroEventReader(string path);    //Constructor prototype
	    ~NeuroEventReader();   //Destructor prototype
	    nEvent_t readEvent();
};
#endif /*NEUROEVENTREADER_H*/
