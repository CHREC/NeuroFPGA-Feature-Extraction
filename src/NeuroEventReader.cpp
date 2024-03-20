/*
Copyright (c) 2024 NSF Center for Space, High-performance, and Resilient Computing (SHREC) University of Pittsburgh. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS AS IS AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#include "../include/NeuroEventReader.hpp"

NeuroEventReader::NeuroEventReader(string path)
{
	// read stream of events
	//tmpFile.open(path, ios::binary);
	evtData = new ifstream(path, ios::binary);
	events = 0;
	prev = 0;
}

NeuroEventReader::~NeuroEventReader()
{
	evtData->close();
}

nEvent_t NeuroEventReader::readEvent()
{
	// init
	unsigned int xPosI = 0, yPosI = 0, timestampI = 0, polI = 0, tsUpper = 0, tsMid = 0, tsLower = 0, xUpper = 0, xLower = 0, yUpper = 0, yLower = 0;
	char polRaw = 0;
	char* xPos;
	char* yPos;
	char* pol = new char[1];
	char* timestamp = new char[2];
	nEvent_t cur_event;
	cur_event.x = WIDTH + 1;
	cur_event.y = HEIGHT + 1;
	cur_event.p = 0;
	cur_event.t = 0;

	// check for EOF
	if(evtData->eof() || events == EVPS)
	{
		evtData->close();
		return cur_event;
	}

	switch(D_SOURCE)
	{
		case 0:
			xPos = new char[1];
			yPos = new char[1];

			evtData->read(xPos, 1);				// first byte is the x event address
			evtData->read(yPos, 1);				// second byte is the y event address

			xPosI = (unsigned int)xPos[0];				// convert values to Int to work with
			yPosI = (unsigned int)yPos[0];

			break;
		case 1:
			xPos = new char[3];

			evtData->read(xPos, 3); // read all 48 bits into memory
/*
			xUpper = (unsigned int) xPos[1];
			xLower = (unsigned int) xPos[0];
			xUpper = xUpper << 8;			// shift into place
			xUpper = xUpper & 0x00000F00;	// mask for 4 bits

			xPosI = ((xUpper) + (xLower));
			xPosI = xPosI & 0x00000FFF;		// never more than 12 bits

			yLower = (unsigned int) xPos[1];
			yUpper = (unsigned int) xPos[2];
			yLower = yLower >> 4; 			//shift into place, remove first 4 bits
			yLower = yLower & 0x0000000F;	//mask for 4 bits
			yUpper = yUpper << 4;			//shift into place, need first 4 bits from lower
			yUpper = yUpper & 0x00000FF0;

			yPosI = ((yUpper) + (yLower));
			yPosI = yPosI & 0x00000FFF;		// never more than 12 bits*/
            
            xUpper = (unsigned int) xPos[0];
            xUpper = xUpper << 4;			// shift into place
            xUpper = xUpper & 0x00000FF0;	// mask for 8 bits
            xLower = (unsigned int) xPos[1];
            xLower = xLower >> 4;			// shift into place
            xLower = xLower & 0x0000000F;	// mask for 4 bits

            xPosI = ((xUpper) + (xLower));
            xPosI = xPosI & 0x00000FFF;		// never more than 12 bits

            yUpper = (unsigned int) xPos[1];
            yUpper = yUpper & 0x0000000F;
            yUpper = yUpper << 8;			//shift into place, need first 4 bits from lower
            yUpper = yUpper & 0x00000F00;
            yLower = (unsigned int) xPos[2];
            yLower =  yLower & 0x0FF;

            yPosI = ((yUpper) + (yLower));
            yPosI = yPosI & 0x00000FFF;		// never more than 12 bits

			break;
		default:
			cout << "Option "<< D_SOURCE <<" source parser not implemented!" << endl;
			break;
	}

	// test against bad data
	if(xPosI > WIDTH || yPosI > HEIGHT)
	{
		cout << "Warning: Bad data stream! x: " << to_string(xPosI) << " y: " << to_string(yPosI) << endl;
		evtData->close();
		return cur_event;
	}

	// read in common formats (polarity, timestamp)
	evtData->read(pol, 1);				// next bit is polarity, read full byte
	evtData->read(timestamp, 2);			// 23 bits for timestamp, read 2 next bytes,

	polRaw = pol[0];						// attempt to separate shifting issues - timestamps from NMNIST data set are out of order
	polI = (polRaw >> 7) & 0x01;			// shift and mask character to get bit

	// unclear exactly what was the previous issue, but timestamps are being created properly 10/10
	// make sure data is shifted properly
	tsUpper = (unsigned int) (pol[0] & 0x7F);
	tsMid = (unsigned int) timestamp[0];
	tsLower = (unsigned int) timestamp[1];

	// mask off polarity and shift 2 bytes, add upperbyte shifted by one byte, add lowest byte
	// data sanitization
	tsUpper = tsUpper << 16;
	tsUpper = tsUpper & 0x007F0000;

	tsMid = tsMid << 8;
	tsMid = tsMid & 0x0000FF00;

	timestampI = ((tsUpper) + (tsMid) + (tsLower)); 	// concat parts, bitwise or returns wierd values, addition works
	timestampI = timestampI & 0x007FFFFF;				// should never be more than 23 bits


	/*if(prev <= timestampI)
	{
		prev = timestampI;
		events++;
	}*/
    
    events++;

	if(timestampI == 8388560)
	{
		prev = 0;
		events--;
	}


	// infile debugging
	//cout << "x: " << to_string(xPosI) << " y: " << to_string(yPosI) << " p: " << to_string(polI) << " t: " << to_string(timestampI) << endl;

	cur_event.x = xPosI;
	cur_event.y = yPosI;
	cur_event.p = polI;
	cur_event.t = timestampI;

	return cur_event;
}
