#ifndef LOSSLAYER_CPP
#define LOSSLAYER_CPP

#include<iostream>
#include<vector>
#include"tensor.cpp"
using namespace std;

class LossLayer{
public:
    LossLayer() : _loss(0.0f) {};
    virtual void forward(Tensor4D &input, Tensor4D &label)=0;
    virtual void backward(Tensor4D &input, Tensor4D &label, Tensor4D &input_grad)=0;
    double loss() { return _loss; }
protected:
    double _loss;
};

class QuadLossLayer : public LossLayer{
public:
    void forward(Tensor4D &input, Tensor4D &label);
    void backward(Tensor4D &input, Tensor4D &label, Tensor4D &input_grad);
};

class CrossEntropyLossLayer : public LossLayer{
public:
    void forward(Tensor4D &input, Tensor4D &label);
    void backward(Tensor4D &input, Tensor4D &label, Tensor4D &input_grad);
    void backward_skip(Tensor4D &input, Tensor4D &label, Tensor4D &input_grad);
};


void QuadLossLayer::forward(Tensor4D &input, Tensor4D &label){
    _loss = 0.0;
    auto pw = [] (double x) { return x * x; };
    for(int i = 0; i < input.size.len1; i++)
        for(int j = 0; j < input.size.len2; j++)
            _loss += pw(input.data[i][j][0][0] - label.data[i][j][0][0]);
    _loss /= 2 * input.size.len1;
}

void QuadLossLayer::backward(Tensor4D &input, Tensor4D &label, Tensor4D &input_grad){
    double B = 0.5 / input_grad.size.len1;
    for(int i = 0; i < input_grad.size.len1; i++)
        for(int j = 0; j < input_grad.size.len2; j++)
            input_grad.data[i][j][0][0] = (input.data[i][j][0][0] - label.data[i][j][0][0]) * B;
}


void CrossEntropyLossLayer::forward(Tensor4D &input, Tensor4D &label){
    _loss = 0.0;
    for(int batch = 0; batch < input.size.len1; ++batch){
        for(int c = 0; c < input.size.len2; ++c){
            _loss += - label.data[batch][c][0][0] * log(input.data[batch][c][0][0]);
        }
    }
    _loss /= input.size.len1;
}

void CrossEntropyLossLayer::backward(Tensor4D &input, Tensor4D &label, Tensor4D &input_grad){
    double B = 1.0 / input.size.len1;
    for(int batch = 0; batch < input_grad.size.len1; ++batch)
        for(int c = 0; c < input_grad.size.len2; ++c)
            input_grad.data[batch][c][0][0] = - B * label.data[batch][c][0][0] / input.data[batch][c][0][0];
}

void CrossEntropyLossLayer::backward_skip(Tensor4D &softmax_out, Tensor4D &label, Tensor4D &softmax_input_grad){
    double B = 1.0 / softmax_input_grad.size.len1;
    for(int batch = 0; batch < softmax_input_grad.size.len1; ++batch)
        for(int c = 0; c < softmax_input_grad.size.len2; ++c)
            softmax_input_grad.data[batch][c][0][0] = B * (softmax_out.data[batch][c][0][0] - label.data[batch][c][0][0]);
}

#endif