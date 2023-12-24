//Numpy array shape [1, 1, 1]
//Min 0.125000000000
//Max 0.250000000000
//Number of zeros 0

#ifndef S18_H_
#define S18_H_

#ifndef __SYNTHESIS__
exponent_scale18_t s18[5];
#else
exponent_scale18_t s18[5] = {{1, -2}, {1, -2}, {1, -2}, {1, -3}, {1, -3}};
#endif

#endif
