#ifndef FLATTENLAYER_CPP
#define FLATTENLAYER_CPP

#include"tensor.cpp"

class FlattenLayer{
public:
    void forward(Tensor4D &input, Tensor4D &output);
    void backward(Tensor4D &output_grad, Tensor4D &input_grad);
};

void FlattenLayer::forward(Tensor4D &input, Tensor4D &output){
    for(int batch = 0; batch < input.size.len1; ++batch){
        for(int c_in = 0; c_in < input.size.len2; ++c_in){
            for(int h_in = 0; h_in < input.size.len3; ++h_in){
                for(int w_in = 0; w_in < input.size.len4; ++w_in){
                    output.data[batch][c_in * input.size.len3 * input.size.len4 + h_in * input.size.len4+ w_in][0][0] = input.data[batch][c_in][h_in][w_in];
                }
            }
        }
    }
}

void FlattenLayer::backward(Tensor4D &output_grad, Tensor4D &input_grad){
    int c_out, h_out, w_out;
    for(int batch = 0; batch < output_grad.size.len1; ++batch){
        for(int c_in = 0; c_in < output_grad.size.len2; ++c_in){
            c_out = c_in / (input_grad.size.len3 * input_grad.size.len4);
            h_out = c_in % (input_grad.size.len3 * input_grad.size.len4) / input_grad.size.len4;
            w_out =  c_in % (input_grad.size.len3 * input_grad.size.len4) % input_grad.size.len4;
            input_grad.data[batch][c_out][h_out][w_out] = output_grad.data[batch][c_in][0][0];
        }
    }
}

#endif