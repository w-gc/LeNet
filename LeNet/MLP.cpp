#ifndef MLP_CPP
#define MLP_CPP

#include"tensor.cpp"

class MLP{
public:
    MLP(TensorShape size, Optimizer *opm);
    TensorShape getWeightSize();
    void printSize();
    void printWeight();
    void printWeight_grad();
    void printMaxMinGrad();
    void setWeight(Tensor4D *new_weight);

    void forward(Tensor4D &input, Tensor4D &output);
    void backward(Tensor4D &input, Tensor4D &output_grad, Tensor4D &input_grad);
    void update();
    void grad_zero();
// private:
    Tensor4D *weight;
    Tensor4D *weight_grad;
    Optimizer *optim;
    // Tensor4D bias;
};

MLP::MLP(TensorShape size, Optimizer *opm) : weight(new Tensor4D(size)), weight_grad(new Tensor4D(size)), optim(opm){
    weight->init_rand();
    weight_grad->init_zero();
}

TensorShape MLP::getWeightSize() {
    return this->weight->size;
}

void MLP::printSize(){
	this->weight->size.print();
}

void MLP::printWeight(){
    weight->printData();
}

void MLP::printMaxMinGrad(){
    cout << "max: " << weight_grad->Max() << ", min: " << weight_grad->Min() << endl;
}

void MLP::printWeight_grad(){
    weight_grad->printData();
}

void MLP::setWeight(Tensor4D *new_weight){
    swap(weight, new_weight);
}

void MLP::forward(Tensor4D &input, Tensor4D &output){
    for(int batch = 0; batch < output.size.len1; ++batch){
        for(int c_out = 0; c_out < output.size.len2; ++c_out){
            for(int c_in = 0; c_in < input.size.len2; ++c_in){
                output.data[batch][c_out][0][0] += input.data[batch][c_in][0][0] * weight->data[c_out][c_in][0][0];
            }
        }
    }
}
void MLP::backward(Tensor4D &input, Tensor4D &output_grad, Tensor4D &input_grad){
    // 计算卷积核的梯度
    for(int c_out = 0; c_out < weight_grad->size.len1; ++c_out){
        for(int c_in = 0; c_in < weight_grad->size.len2; ++c_in){
            // ------------------------------------------------------------------------------------------
            for(int batch = 0; batch < input_grad.size.len1; ++batch){
                weight_grad->data[c_out][c_in][0][0] += input.data[batch][c_in][0][0] * output_grad.data[batch][c_out][0][0];
            }
            // ------------------------------------------------------------------------------------------
        }
    }
    
    // 误差返回到上一层
    for(int batch = 0; batch < input_grad.size.len1; ++batch){
        for(int c_in = 0; c_in < input_grad.size.len2; ++c_in){
            // ------------------------------------------------------------------------------------------
            for(int c_out = 0; c_out < output_grad.size.len2; ++c_out){
                input_grad.data[batch][c_in][0][0] += weight->data[c_out][c_in][0][0] * output_grad.data[batch][c_out][0][0];
            }
            // ------------------------------------------------------------------------------------------
        }
    }
}

void MLP::update(){
    optim->update(*weight, *weight_grad);
}

void MLP::grad_zero(){
    weight_grad->init_zero();
}

#endif