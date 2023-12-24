#include <iostream>

#include "myproject.h"
#include "parameters.h"

void myproject(
    hls::stream<input_t> &conv2d1_input,
    hls::stream<result_t> &layer16_out
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS INTERFACE axis port=conv2d1_input,layer16_out 
    #pragma HLS DATAFLOW 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<weight2_t, 675>(w2, "w2.txt");
        nnet::load_weights_from_txt<bias2_t, 25>(b2, "b2.txt");
        nnet::load_exponent_weights_from_txt<exponent_scale17_t, 25>(s17, "s17.txt");
        nnet::load_weights_from_txt<bias17_t, 25>(b17, "b17.txt");
        nnet::load_weights_from_txt<weight6_t, 1125>(w6, "w6.txt");
        nnet::load_weights_from_txt<bias6_t, 5>(b6, "b6.txt");
        nnet::load_exponent_weights_from_txt<exponent_scale18_t, 5>(s18, "s18.txt");
        nnet::load_weights_from_txt<bias18_t, 5>(b18, "b18.txt");
        nnet::load_weights_from_txt<weight11_t, 19600>(w11, "w11.txt");
        nnet::load_weights_from_txt<bias11_t, 5>(b11, "b11.txt");
        nnet::load_weights_from_txt<weight14_t, 15>(w14, "w14.txt");
        nnet::load_weights_from_txt<bias14_t, 3>(b14, "b14.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    hls::stream<layer2_t> layer2_out("layer2_out");
    #pragma HLS STREAM variable=layer2_out depth=900
    nnet::conv_2d_cl<input_t, layer2_t, config2>(conv2d1_input, layer2_out, w2, b2); // conv2d1

    hls::stream<layer17_t> layer17_out("layer17_out");
    #pragma HLS STREAM variable=layer17_out depth=900
    nnet::normalize<layer2_t, layer17_t, config17>(layer2_out, layer17_out, s17, b17); // conv2d1_alpha

    hls::stream<layer3_t> layer3_out("layer3_out");
    #pragma HLS STREAM variable=layer3_out depth=900
    nnet::linear<layer17_t, layer3_t, linear_config3>(layer17_out, layer3_out); // conv2d1_linear

    hls::stream<layer4_t> layer4_out("layer4_out");
    #pragma HLS STREAM variable=layer4_out depth=900
    nnet::relu<layer3_t, layer4_t, relu_config4>(layer3_out, layer4_out); // relu1

    hls::stream<layer5_t> layer5_out("layer5_out");
    #pragma HLS STREAM variable=layer5_out depth=900
    nnet::pooling2d_cl<layer4_t, layer5_t, config5>(layer4_out, layer5_out); // max_pooling2d

    hls::stream<layer6_t> layer6_out("layer6_out");
    #pragma HLS STREAM variable=layer6_out depth=784
    nnet::conv_2d_cl<layer5_t, layer6_t, config6>(layer5_out, layer6_out, w6, b6); // conv2d2

    hls::stream<layer18_t> layer18_out("layer18_out");
    #pragma HLS STREAM variable=layer18_out depth=784
    nnet::normalize<layer6_t, layer18_t, config18>(layer6_out, layer18_out, s18, b18); // conv2d2_alpha

    hls::stream<layer7_t> layer7_out("layer7_out");
    #pragma HLS STREAM variable=layer7_out depth=784
    nnet::linear<layer18_t, layer7_t, linear_config7>(layer18_out, layer7_out); // conv2d2_linear

    hls::stream<layer8_t> layer8_out("layer8_out");
    #pragma HLS STREAM variable=layer8_out depth=784
    nnet::relu<layer7_t, layer8_t, relu_config8>(layer7_out, layer8_out); // relu2

    hls::stream<layer9_t> layer9_out("layer9_out");
    #pragma HLS STREAM variable=layer9_out depth=784
    nnet::pooling2d_cl<layer8_t, layer9_t, config9>(layer8_out, layer9_out); // max_pooling2d_1

    auto& layer10_out = layer9_out;
    hls::stream<layer11_t> layer11_out("layer11_out");
    #pragma HLS STREAM variable=layer11_out depth=1
    nnet::dense<layer9_t, layer11_t, config11>(layer10_out, layer11_out, w11, b11); // fc1

    hls::stream<layer13_t> layer13_out("layer13_out");
    #pragma HLS STREAM variable=layer13_out depth=1
    nnet::relu<layer11_t, layer13_t, relu_config13>(layer11_out, layer13_out); // relu3

    hls::stream<layer14_t> layer14_out("layer14_out");
    #pragma HLS STREAM variable=layer14_out depth=1
    nnet::dense<layer13_t, layer14_t, config14>(layer13_out, layer14_out, w14, b14); // fc2

    nnet::softmax<layer14_t, result_t, softmax_config16>(layer14_out, layer16_out); // activation

}
