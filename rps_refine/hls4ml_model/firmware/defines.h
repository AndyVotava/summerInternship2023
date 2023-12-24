#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 32
#define N_INPUT_2_1 32
#define N_INPUT_3_1 3
#define OUT_HEIGHT_2 30
#define OUT_WIDTH_2 30
#define N_FILT_2 25
#define OUT_HEIGHT_2 30
#define OUT_WIDTH_2 30
#define N_FILT_2 25
#define OUT_HEIGHT_2 30
#define OUT_WIDTH_2 30
#define N_FILT_2 25
#define OUT_HEIGHT_2 30
#define OUT_WIDTH_2 30
#define N_FILT_2 25
#define OUT_HEIGHT_5 30
#define OUT_WIDTH_5 30
#define N_FILT_5 25
#define OUT_HEIGHT_6 28
#define OUT_WIDTH_6 28
#define N_FILT_6 5
#define OUT_HEIGHT_6 28
#define OUT_WIDTH_6 28
#define N_FILT_6 5
#define OUT_HEIGHT_6 28
#define OUT_WIDTH_6 28
#define N_FILT_6 5
#define OUT_HEIGHT_6 28
#define OUT_WIDTH_6 28
#define N_FILT_6 5
#define OUT_HEIGHT_9 28
#define OUT_WIDTH_9 28
#define N_FILT_9 5
#define N_SIZE_0_10 3920
#define N_LAYER_11 5
#define N_LAYER_11 5
#define N_LAYER_14 3
#define N_LAYER_14 3

// hls-fpga-machine-learning insert layer-precision
typedef nnet::array<ap_fixed<16,6>, 3*1> input_t;
typedef ap_fixed<7,1> model_default_t;
typedef nnet::array<ap_fixed<16,6>, 25*1> layer2_t;
typedef ap_fixed<7,2> weight2_t;
typedef ap_fixed<7,2> bias2_t;
typedef nnet::array<ap_fixed<7,1>, 25*1> layer17_t;
typedef struct exponent_scale17_t {ap_uint<1> sign;ap_int<18> weight; } exponent_scale17_t;
typedef ap_fixed<7,2> bias17_t;
typedef nnet::array<ap_fixed<16,6>, 25*1> layer3_t;
typedef ap_fixed<18,8> conv2d1_linear_table_t;
typedef nnet::array<ap_ufixed<7,0,AP_RND_CONV,AP_SAT>, 25*1> layer4_t;
typedef ap_fixed<18,8> relu1_table_t;
typedef nnet::array<ap_fixed<16,6>, 25*1> layer5_t;
typedef nnet::array<ap_fixed<16,6>, 5*1> layer6_t;
typedef ap_fixed<7,2> weight6_t;
typedef ap_fixed<7,2> bias6_t;
typedef nnet::array<ap_fixed<7,1>, 5*1> layer18_t;
typedef struct exponent_scale18_t {ap_uint<1> sign;ap_int<4> weight; } exponent_scale18_t;
typedef ap_fixed<7,2> bias18_t;
typedef nnet::array<ap_fixed<16,6>, 5*1> layer7_t;
typedef ap_fixed<18,8> conv2d2_linear_table_t;
typedef nnet::array<ap_ufixed<7,0,AP_RND_CONV,AP_SAT>, 5*1> layer8_t;
typedef ap_fixed<18,8> relu2_table_t;
typedef nnet::array<ap_fixed<16,6>, 5*1> layer9_t;
typedef nnet::array<ap_fixed<16,6>, 5*1> layer11_t;
typedef ap_fixed<7,2> weight11_t;
typedef ap_fixed<7,2> bias11_t;
typedef ap_uint<1> layer11_index;
typedef nnet::array<ap_ufixed<7,0,AP_RND_CONV,AP_SAT>, 5*1> layer13_t;
typedef ap_fixed<18,8> relu3_table_t;
typedef nnet::array<ap_fixed<16,6>, 3*1> layer14_t;
typedef ap_fixed<7,2> weight14_t;
typedef ap_fixed<7,2> bias14_t;
typedef ap_uint<1> layer14_index;
typedef nnet::array<ap_fixed<16,6>, 3*1> result_t;
typedef ap_fixed<18,8> activation_table_t;
typedef ap_fixed<18,8,AP_RND,AP_SAT> activation_exp_table_t;
typedef ap_fixed<18,8,AP_RND,AP_SAT> activation_inv_table_t;

#endif
