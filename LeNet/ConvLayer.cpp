#ifndef CONVLAYER_CPP
#define CONVLAYER_CPP

#include<iostream>
#include"tensor.cpp"
#include"Optimizer.cpp"
using namespace std;

class ConvLayer{
public:
    ConvLayer(TensorShape weight_size, int s, Optimizer *opm);
    // ~ConvLayer();
    TensorShape getWeightSize();
    void printSize();
    void printWeight();
    void printWeight_grad();
    void printMaxMinGrad();
    void setWeight(Tensor4D *new_weight);

    void forward(Tensor4D &input, Tensor4D &output);
    void backward(Tensor4D &input, Tensor4D &output, Tensor4D &input_grad);
    void update();
    void grad_zero();
// private:
    Tensor4D *weight;
    Tensor4D *weight_grad;
    Optimizer *optim;

    // Tensor4D bias;
    int stride;
};

ConvLayer::ConvLayer(TensorShape weight_size, int s, Optimizer *opm) : 
        weight(new Tensor4D(weight_size)), weight_grad(new Tensor4D(weight_size)), stride(s), optim(opm){
    weight->init_rand();
    weight_grad->init_zero();
}

// ConvLayer::~ConvLayer(){
//     delete weight;
//     delete weight_grad;
//     cout << "test" << endl;
// }

TensorShape ConvLayer::getWeightSize() {
    return this->weight->size;
}

void ConvLayer::printSize(){
	this->weight->size.print();
}

void ConvLayer::printWeight(){
    weight->printData();
}

void ConvLayer::printWeight_grad(){
    weight_grad->printData();
}

void ConvLayer::printMaxMinGrad(){
    cout << "max: " << weight_grad->Max() << ", min: " << weight_grad->Min() << endl;
}

void ConvLayer::setWeight(Tensor4D *new_weight){
    swap(weight, new_weight);
}

void ConvLayer::forward(Tensor4D &input, Tensor4D &output){
    for(int batch = 0; batch < output.size.len1; ++batch){
        for(int c_out = 0; c_out < output.size.len2; ++c_out){
            // output[batch][c_out][][] += weight[c_out][c_in][][] conv input[batch][c_in][][] 
            for(int h_out = 0; h_out < output.size.len3; ++h_out){
                for(int w_out = 0; w_out < output.size.len4; ++w_out){
                // ------------------------------------------------------------------------------------------
                    // cout << "output: (" << batch << ", " << c_out << ", " << h_out << ", " << w_out << ")" << endl;
                    for(int c_in = 0; c_in < weight->size.len2; ++c_in){
                        for(int h_f = 0; h_f < weight->size.len3; ++h_f){
                            for(int w_f = 0; w_f < weight->size.len4; ++w_f){
                                // cout << "filter: (" << c_out << ", " << c_in << ", " << h_f << ", " << w_f << ")" << endl;
                                // cout << "input: (" << batch << ", " << c_in << ", " << h_f + stride * h_out << ", " << w_f + stride * w_out << ")" << endl;
                                output.data[batch][c_out][h_out][w_out] += weight->data[c_out][c_in][h_f][w_f] * input.data[batch][c_in][stride * h_out + h_f][stride * w_out + w_f];
                            }
                        }
                    }
                // ------------------------------------------------------------------------------------------
                }
            }
        }
    }
}

void ConvLayer::backward(Tensor4D &input, Tensor4D &output_grad, Tensor4D &input_grad){
    // calculate the grad of the weights of conv 
    for(int c_out = 0; c_out < weight->size.len1; ++c_out){
        for(int c_in = 0; c_in < weight->size.len2; ++c_in){
            for(int h_f = 0; h_f < weight->size.len3; ++h_f){
                for(int w_f = 0; w_f < weight->size.len4; ++w_f){
                // ------------------------------------------------------------------------------------------
                    for(int batch = 0; batch < input.size.len1; ++batch){
                        for(int h_out = 0; h_out < output_grad.size.len3; ++h_out){
                            for(int w_out = 0; w_out < output_grad.size.len4; ++w_out){
                                weight_grad->data[c_out][c_in][h_f][w_f] += output_grad.data[batch][c_out][h_out][w_out] * input.data[batch][c_in][stride*h_out+h_f][stride*w_out+w_f];
                            }
                        }
                    }
                // ------------------------------------------------------------------------------------------
                }
            }
        }
    }


    int h_f = weight->size.len3;
    int w_f = weight->size.len4;
    double s = stride;
    // calculate the grad of the conv's input data
    for(int batch = 0; batch < input_grad.size.len1; ++batch){
        for(int c_in = 0; c_in < input_grad.size.len2; ++c_in){
            for(int h_in = 0; h_in < input_grad.size.len3; ++h_in){
                    for(int w_in = 0; w_in < input_grad.size.len4; ++w_in){
                    // ------------------------------------------------------------------------------------------
                        for(int c_out = 0; c_out < output_grad.size.len2; ++c_out){
                            for(int h_out = max(ceil((h_in - h_f + 1) / s), 0.0); h_out <= min(floor(h_in / s), output_grad.size.len3 - 1.0); ++h_out){
                                for(int w_out = max(ceil((w_in - w_f + 1) / s), 0.0); w_out <= min(floor(w_in / s), output_grad.size.len4 - 1.0); ++w_out){
                                    input_grad.data[batch][c_in][h_in][w_in] += output_grad.data[batch][c_out][h_out][w_out] * weight->data[c_out][c_in][h_in - stride*h_out][w_in - stride*w_out];
                                }
                            }
                        }
                    // ------------------------------------------------------------------------------------------
                }
            }
        }
    }

}

void ConvLayer::update(){
    optim->update(*weight, *weight_grad);
}

void ConvLayer::grad_zero(){
    weight_grad->init_zero();
}

#endif