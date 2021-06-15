#ifndef ACTILAYER_CPP
#define ACTILAYER_CPP

#include"tensor.cpp"
#include"utilize.cpp"
#include<vector>
#include<cmath>

class ActiLayer{
public:
    virtual void forward(Tensor4D &input, Tensor4D &output)=0;
    virtual void backward(Tensor4D &input, Tensor4D &output_grad, Tensor4D &input_grad)=0;
};

class SigmoidLayer : public ActiLayer{
public:
    void forward(Tensor4D &input, Tensor4D &output){
        input.pointwiseFun(output, Sigmoid<double>());
    };
    void backward(Tensor4D &input, Tensor4D &output_grad, Tensor4D &input_grad){ 
        input.pointwiseFun(input_grad, Sigmoid_back<double>());
        input_grad.TensorOpTensor_(output_grad, op_mul<double>());
    };
};

class TanhLayer : public ActiLayer{
public:
    void forward(Tensor4D &input, Tensor4D &output){ 
        input.pointwiseFun(output, Tanh<double>()); 
    };
    void backward(Tensor4D &input, Tensor4D &output_grad, Tensor4D &input_grad){ 
        input.pointwiseFun(input_grad, Tanh_back<double>());
        input_grad.TensorOpTensor_(output_grad, op_mul<double>());
    };
};

class ReLULayer : public ActiLayer{
public:
    void forward(Tensor4D &input, Tensor4D &output){ 
        input.pointwiseFun(output, ReLU<double>()); 
    }
    void backward(Tensor4D &input, Tensor4D &output_grad, Tensor4D &input_grad){
        // input.pointwiseFun_(ReLU_back<double>());
        // input_grad.TensorOpTensor(input, output_grad, op_mul<double>());
        input.pointwiseFun(input_grad, ReLU_back<double>());
        input_grad.TensorOpTensor_(output_grad, op_mul<double>());
    }
};

class SoftMaxLayer : public ActiLayer{
public:
    void forward(Tensor4D &input, Tensor4D &output);
    void backward(Tensor4D &output, Tensor4D &output_grad, Tensor4D &input_grad);
};

void SoftMaxLayer::forward(Tensor4D &input, Tensor4D &output){ 
    for(int batch = 0; batch < input.size.len1; ++batch){
        double max_num = input.data[batch][0][0][0];
        double sum = 0.0;
        for(int c = 0; c < input.size.len2; ++c){
            max_num = max(max_num, input.data[batch][c][0][0]);
        }
        for(int c = 0; c < output.size.len2; ++c){
            output.data[batch][c][0][0] = exp(input.data[batch][c][0][0] - max_num);
            sum += output.data[batch][c][0][0];
        }
        for(int c = 0; c < output.size.len2; ++c){
            output.data[batch][c][0][0] /= sum;
        }
    }
};

void SoftMaxLayer::backward(Tensor4D &output, Tensor4D &output_grad, Tensor4D &input_grad){ 
    for(int batch = 0; batch < input_grad.size.len1; ++batch){
        for(int c_in = 0; c_in < input_grad.size.len2; ++c_in){
            for(int c_out = 0; c_out < output_grad.size.len2; ++c_out){
                input_grad.data[batch][c_in][0][0] += - output_grad.data[batch][c_out][0][0] * output.data[batch][c_out][0][0] * output.data[batch][c_in][0][0];
                if(c_out == c_in) input_grad.data[batch][c_in][0][0] += output_grad.data[batch][c_out][0][0] * output.data[batch][c_out][0][0];
            }
        }
    }
};


#endif