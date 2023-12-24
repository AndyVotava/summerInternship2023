//Numpy array shape [5]
//Min -0.031250000000
//Max 0.031250000000
//Number of zeros 3

#ifndef B18_H_
#define B18_H_

#ifndef __SYNTHESIS__
bias18_t b18[5];
#else
bias18_t b18[5] = {0.00000, 0.03125, 0.00000, 0.00000, -0.03125};
#endif

#endif
