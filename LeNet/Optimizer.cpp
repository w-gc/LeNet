#ifndef OPTIMIZER_CPP
#define OPTIMIZER_CPP

#include<iostream>
#include"tensor.cpp"
using namespace std;

class Optimizer{
public:
    double lr;
    Optimizer(): lr(-0.01f) {};
    Optimizer(double learn_rate) : lr(learn_rate){};
    virtual void update(double &weight, double &grad)=0;
    virtual void update(Tensor4D &weight, Tensor4D &grad)=0; // { cout << "TODO"  << endl; exit(-1); };
    // virtual ~Optimizer();
};

class SGD: public Optimizer{
public:
    SGD() : Optimizer(-0.01f), weight_decay(1 - 1e-4) {};
    SGD(double lr, double wd) : Optimizer(-lr), weight_decay(1 - wd) {};
    void update(double &weight, double &grad);
    void update(Tensor4D &weight, Tensor4D &grad);
    void operator()(Tensor4D &weight, Tensor4D &grad){ this->update(weight, grad); }
    double weight_decay;
};

void SGD::update(double &weight, double &grad){
    weight = this->weight_decay * weight + this->lr * grad;
}

void SGD::update(Tensor4D &weight, Tensor4D &grad){
    grad.scalarOpTensor_(this->lr, op_mul<double>());
    // weight.scalarOpTensor_(this->weight_decay, op_mul<double>());
    weight.TensorOpTensor_(grad, op_plus<double>());
}

#endif