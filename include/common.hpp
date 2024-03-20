#ifndef COMMON_H
#define COMMON_H

//modified so FPGA can use it (no iostream, string, std)

#define RAD 5			// radius of time-context
#define TAU 800 			// time constant tau (ms)
#define NF 16 			// number of features
#define KR 2 			// R scaling factor
#define KT 10 			// T scaling factor
#define KN 2 			// N scaling factor
#define LAYERS 3		// number of layers
#define WIDTH 640		// input sensor Height
#define HEIGHT 480		// input sensor Width
#define CLASSES 5		// number of classes
#define SPC 1   		// number of training samples per class
#define EVPS 100		// events per sample
#define TCB 100         // time context buffer size
#define DIMS 2			// data dimentionality for kmeans (x, y)
#define DIMS0 121         // number of points in layer 0 timesurface
#define DIMS1 441         // number of points in layer 1 timesurface
#define DIMS2 1681        // number of points in layer 2 timesurface
#define H_ALGO 1 		// algorithm for histogram classifications (0 = standard distance, 1 = normalized distance, 2 = Bhattacharyya distance)
#define C_ALGO 1		// algorithm for classification (0 = histogram, 1 = perceptron)
#define D_SOURCE 0 		// data source specifier (0 = 8 bit x bit, 1 = 9 bit x 9 bit)
#define F_FEAT 64       // final number of features

#endif /* common */
