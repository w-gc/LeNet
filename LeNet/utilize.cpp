#ifndef UTILIZE_CPP
#define UTILIZE_CPP

#include<cmath>

template<typename T>
struct op_plus{
    T operator()(T a, T b){return a + b;}
};

template<typename T>
struct op_minus{
    T operator()(T a, T b){return a - b;}
};

template<typename T>
struct op_mul{
    T operator()(T a, T b){return a * b;}
};

template<typename T>
struct op_div{
    T operator()(T a, T b){return a / b;}
};

template<typename T>
struct op_assign{
    T operator()(T a, T b){return a = b;}
};

template<typename T>
struct op_exp{
    T operator()(T x){return exp(x);}
};

template<typename T>
struct op_log{
    T operator()(T x){return log(x);}
};

template<typename T, typename N>
struct op_power{
    T operator()(T x, N n){ return pow(x, n);}
};


template<typename T>
struct Sigmoid{
    T operator()(T &x){return 1.0 / (1.0 + exp(-x));}
};

template<typename T>
struct Sigmoid_back{
    T operator()(T &x){
        T f_x = Sigmoid<T>()(x);
        return f_x * (1.0 - f_x);
    }
};

template<typename T>
struct Tanh{
    T operator()(T &x){
        T exp_x= exp(x), exp_x_ = exp(-x); 
        return (exp_x + exp_x_) / (exp_x - exp_x_);
    }
};

template<typename T>
struct Tanh_back{
    T operator()(T &x){
        T f_x= Tanh<T>()(x);
        return 1 - f_x * f_x;
    }
};

template<typename T>
struct ReLU{
    T operator()(T &x){ return x > 0.0 ? x : 0.0; }
};

template<typename T>
struct ReLU_back{
    T operator()(T &x){ return x > 0.0 ? 1.0 : 0.0;}
};

/*
template<typename T>
struct op_plus{
    T operator()(T &a, T &b){return a + b;}
};

template<typename T>
struct op_minus{
    T operator()(T &a, T &b){return a - b;}
};

template<typename T>
struct op_mul{
    T operator()(T &a, T &b){return a * b;}
};

template<typename T>
struct op_div{
    T operator()(T &a, T &b){return a / b;}
};

template<typename T>
struct op_assign{
    T operator()(T &a, T &b){return a = b;}
};

template<typename T>
struct op_exp{
    T operator()(T &x){return exp(x);}
};

template<typename T>
struct op_log{
    T operator()(T &x){return log(x);}
};

template<typename T, typename N>
struct op_power{
    T operator()(T &x, N n){return pow(x, n);}
};
*/

#endif