#ifndef POOLLAYER_CPP
#define POOLLAYER_CPP

#include<iostream>
#include<unordered_map>
#include"tensor.cpp"
using namespace std;

struct Location{
    int b, c, h, w;
    bool operator== (const Location &a) const { return a.b == b && a.c == c && a.h == h && a.w == w; }
};

struct HashFun{
    std::size_t operator()(const Location &a) const { return a.b * a.c * a.h * a.w; }
};



class PoolLayer{
public:
    PoolLayer(int h, int w, int s):height(h), width(w), stride(s) {};
    void printSize();

    virtual void forward(Tensor4D &input, Tensor4D &output)=0;
    virtual void backward(Tensor4D &output_grad, Tensor4D &input_grad)=0;
// protected:
    int height;
    int width;
    int stride;
};

class MaxPool : public PoolLayer{
public:
    MaxPool(int h, int w, int s) : PoolLayer(h, w, s) {};
    void forward(Tensor4D &input, Tensor4D &output);
    void backward(Tensor4D &output_grad, Tensor4D &input_grad);
private:
    unordered_map<Location, Location, HashFun> pool_map;
};

class AvgPool : public PoolLayer{
public:
    AvgPool(int h, int w, int s) : PoolLayer(h, w, s) {};
    void forward(Tensor4D &input, Tensor4D &output);
    void backward(Tensor4D &output_grad, Tensor4D &input_grad);
};

void PoolLayer::printSize(){
	cout << "pool size: (" << height << "," << width << ")" << endl;
}

void MaxPool::forward(Tensor4D &input, Tensor4D &output){
    pool_map.clear();
    for(int batch = 0; batch < output.size.len1; ++batch){
        for(int c_out = 0; c_out < output.size.len2; ++c_out){
            for(int h_out = 0; h_out < output.size.len3; ++h_out){
                for(int w_out = 0; w_out < output.size.len4; ++w_out){
                    int res = -1e9;
                    int idx_h, idx_w;
                    for(int h_p = 0; h_p < height; ++h_p){
                        for(int w_p = 0; w_p < width; ++w_p){
                            idx_h = stride * h_out + h_p;
                            idx_w = stride * w_out + w_p;
                            if(res < input.data[batch][c_out][idx_h][idx_w]){
                                res = input.data[batch][c_out][idx_h][idx_w];
                            }
                        }
                    }
                    output.data[batch][c_out][h_out][w_out] = res;
                    pool_map.insert({Location{batch, c_out, h_out, w_out}, Location{batch, c_out, idx_h, idx_w}});

                }
            }
        }
    }
}

void MaxPool::backward(Tensor4D &output_grad, Tensor4D &input_grad){
    for(auto idx : pool_map){
        input_grad.data[idx.second.b][idx.second.c][idx.second.h][idx.second.w] = output_grad.data[idx.first.b][idx.first.c][idx.first.h][idx.first.w];
    }
}


void AvgPool::forward(Tensor4D &input, Tensor4D &output){
    double HW = 1.0 / (height * width);
    for(int batch = 0; batch < output.size.len1; ++batch){
        for(int c_out = 0; c_out < output.size.len2; ++c_out){
            for(int h_out = 0; h_out < output.size.len3; ++h_out){
                for(int w_out = 0; w_out < output.size.len4; ++w_out){
                    for(int h_p = 0; h_p < height; ++h_p){
                        for(int w_p = 0; w_p < width; ++w_p){
                            output.data[batch][c_out][h_out][w_out] += HW* input.data[batch][c_out][stride * h_out + h_p][stride * w_out + w_p];
                        }
                    }
                }
            }
        }
    }
}

void AvgPool::backward(Tensor4D &output_grad, Tensor4D &input_grad){
    double HW = 1.0 / (height * width);
    double s = stride;
    for(int batch = 0; batch < input_grad.size.len1; ++batch){
        for(int c_in = 0; c_in < input_grad.size.len2; ++c_in){
            for(int h_in = 0; h_in < input_grad.size.len3; ++h_in){
                for(int w_in = 0; w_in < input_grad.size.len4; ++w_in){
                    // ------------------------------------------------------------------------------------------
                    for(int h_out = max(ceil((h_in - height + 1) / s), 0.0); h_out <= min(floor(h_in / s), output_grad.size.len3 - 1.0); ++h_out){
                        for(int w_out = max(ceil((w_in - width + 1) / s), 0.0); w_out <= min(floor(w_in / s), output_grad.size.len4 - 1.0); ++w_out){
                            input_grad.data[batch][c_in][h_in][w_in] += output_grad.data[batch][c_in][h_out][w_out] * HW;
                        }
                    }
                    // ------------------------------------------------------------------------------------------
                }
            }
        }
    }
}

#endif