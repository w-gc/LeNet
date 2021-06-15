#ifndef NORMLAYER_CPP
#define NORMLAYER_CPP

#include"tensor.cpp"
#include"utilize.cpp"
#include"Optimizer.cpp"
#include<cmath>

class NormLayer{
public:
    virtual void forward(Tensor4D &input, Tensor4D &output)=0;
    virtual void backward(Tensor4D &input, Tensor4D &output_grad, Tensor4D &input_grad)=0;
};

class BatchNormLayer :  public NormLayer{
public:
    BatchNormLayer(int C, Optimizer *opm) 
        : Mean(new Tensor4D(TensorShape{C,1,1,1})), Var(new Tensor4D(TensorShape{C,1,1,1})), 
        gamma(new Tensor4D(TensorShape{C,1,1,1})), beta(new Tensor4D(TensorShape{C,1,1,1})), 
        gamma_grad(new Tensor4D(TensorShape{C,1,1,1})), beta_grad(new Tensor4D(TensorShape{C,1,1,1})), optim(opm){
            for(int c = 0; c < C; ++c){
                Mean->data[c][0][0][0] = 0.0;
                Var->data[c][0][0][0] = 0.0;
                gamma->data[c][0][0][0] = 1.0;
                beta->data[c][0][0][0] = 0.0;
                gamma_grad->data[c][0][0][0] = 0.0;
                beta_grad->data[c][0][0][0] = 0.0;
            }
    }
    void layer_zero();
    void grad_zero();
    void forward(Tensor4D &input, Tensor4D &output);
    void backward(Tensor4D &output, Tensor4D &output_grad, Tensor4D &input_grad);
    void update();

    Tensor4D *Mean;
    Tensor4D *Var;
    Tensor4D *gamma;
    Tensor4D *beta;
    Tensor4D *gamma_grad;
    Tensor4D *beta_grad;
    Tensor4D *input_data;
    Optimizer *optim;
};

void BatchNormLayer::layer_zero(){
	Mean->init_zero();
	Var->init_zero();
}
void BatchNormLayer::grad_zero(){
	gamma_grad->init_zero();
	beta_grad->init_zero();
}

void BatchNormLayer::forward(Tensor4D &input, Tensor4D &output){
    double BHW = 1.0 / (input.size.len1 * input.size.len3 * input.size.len4);
    for(int b = 0; b < input.size.len1; ++b)
        for(int c = 0; c < input.size.len2; ++c)
            for(int h = 0; h < input.size.len3; ++h)
                for(int w = 0; w < input.size.len4; ++w)
                    Mean->data[c][0][0][0] += input.data[b][c][h][w] * BHW;
    
    for(int b = 0; b < input.size.len1; ++b)
        for(int c = 0; c < input.size.len2; ++c)
            for(int h = 0; h < input.size.len3; ++h)
                for(int w = 0; w < input.size.len4; ++w)
                    Var->data[c][0][0][0] += pow(input.data[b][c][h][w] - Mean->data[c][0][0][0], 2) * BHW;
    
    for(int b = 0; b < input.size.len1; ++b)
        for(int c = 0; c < input.size.len2; ++c)
            for(int h = 0; h < input.size.len3; ++h)
                for(int w = 0; w < input.size.len4; ++w)
                    output.data[b][c][h][w] = gamma->data[c][0][0][0] * (input.data[b][c][h][w] - Mean->data[c][0][0][0]) / pow(Var->data[c][0][0][0] + 1e-5, 0.5) + beta->data[c][0][0][0];
    
    input_data = &input;
};

void BatchNormLayer::backward(Tensor4D &output, Tensor4D &output_grad, Tensor4D &input_grad){ 
    // for(int c = 0; c < input_grad.size.len2; ++c){
    //     gamma_grad->data[c][0][0][0] = 0.0;
    //     beta_grad->data[c][0][0][0] = 0.0;
    // }
    double *var_grad = new double[input_grad.size.len2];
    double *mean_grad = new double[input_grad.size.len2];
    double BHW = 1.0 / (input_grad.size.len1 * input_grad.size.len3 * input_grad.size.len4);

    for(int c = 0; c < input_grad.size.len2; ++c){
        var_grad[c] = 0.0;
        mean_grad[c] = 0.0;
        double tmp = - gamma->data[c][0][0][0] / ( 2 * pow(Var->data[c][0][0][0] + 1e-5, 1.5));
        for(int b = 0; b < input_grad.size.len1; ++b)
            for(int h = 0; h < input_grad.size.len3; ++h)
                for(int w = 0; w < input_grad.size.len4; ++w)
                    var_grad[c] += tmp * output_grad.data[b][c][h][w] * (input_data->data[b][c][h][w] - Mean->data[c][0][0][0]);

        tmp = - gamma->data[c][0][0][0] / pow(Var->data[c][0][0][0] + 1e-5, 0.5);
        for(int b = 0; b < input_grad.size.len1; ++b)
            for(int h = 0; h < input_grad.size.len3; ++h)
                for(int w = 0; w < input_grad.size.len4; ++w)
                    mean_grad[c] += tmp * output_grad.data[b][c][h][w] - 2 * BHW * var_grad[c] * (input_data->data[b][c][h][w] - Mean->data[c][0][0][0]);
    }
    
    for(int b = 0; b < input_grad.size.len1; ++b){
        for(int c = 0; c < input_grad.size.len2; ++c){
            for(int h = 0; h < input_grad.size.len3; ++h){
                for(int w = 0; w < input_grad.size.len4; ++w){
                    gamma_grad->data[c][0][0][0] += output_grad.data[b][c][h][w] * (output.data[b][c][h][w] - beta->data[c][0][0][0]) / gamma->data[c][0][0][0];
                    beta_grad->data[c][0][0][0] += output_grad.data[b][c][h][w];
                    input_grad.data[b][c][h][w] = gamma->data[c][0][0][0] / pow(Var->data[c][0][0][0] + 1e-5, 0.5) * output_grad.data[b][c][h][w] + 2 * BHW * var_grad[c] * (input_data->data[b][c][h][w] - Mean->data[c][0][0][0]) + BHW * mean_grad[c];
                }
            }
        }
    }
};

void BatchNormLayer::update(){
    optim->update(*gamma, *gamma_grad);
    optim->update(*beta, *beta_grad);
}

#endif