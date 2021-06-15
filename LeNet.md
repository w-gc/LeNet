

# ✅ 卷积CONV

## ▶ 前向传播

- 输入$X \in \mathbb{R}^{B \times C_{in} \times H_{in} \times W_{in}}$
  
- 卷积核为$F \in \mathbb{R}^{C_{out} \times C_{in} \times H_{f} \times W_{f}}$，步长为 $s$

- 输出 $Y \in \mathbb{R}^{B \times C_{out} \times H_{out} \times W_{out}}$

- 传播：
$$
Y_{b,c_o, h_o, w_o} = \sum_{c_i=1}^{C_{in}} \sum_{h_f=1}^{H_f} \sum_{w_f=1}^{W_{f}} F_{c_o, c_i, h_f, w_f} X_{b, c_i, s(h_o-1)+h_f, s(w_o-1)+w_f}
$$

```c++
void ConvLayer::forward(Tensor4D &input, Tensor4D &output){
    for(int batch = 0; batch < output.size.len1; ++batch){
        for(int c_out = 0; c_out < output.size.len2; ++c_out){
            for(int h_out = 0; h_out < output.size.len3; ++h_out){
                for(int w_out = 0; w_out < output.size.len4; ++w_out){
                // ------------------------------------------------------------------------------------------
                    for(int c_in = 0; c_in < weight->size.len2; ++c_in){
                        for(int h_f = 0; h_f < weight->size.len3; ++h_f){
                            for(int w_f = 0; w_f < weight->size.len4; ++w_f){
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

```

## ◀ 反向传播

- 输入$\frac{\partial L}{\partial Y} \in \mathbb{R}^{B \times C_{out} \times H_{out} \times W_{out}}$，$X \in \mathbb{R}^{B \times C_{in} \times H_{in} \times W_{in}}$和$F \in \mathbb{R}^{C_{out} \times C_{in} \times H_{f} \times W_{f}}$
  
- 输出 $\frac{\partial L}{\partial X} \in \mathbb{R}^{B \times C_{in} \times H_{in} \times W_{in}}$ 和 $\frac{\partial L}{\partial F} \in \mathbb{R}^{B \times C_{out} \times H_{out} \times W_{out}}$
  
- 反向传播(对卷积核求偏导)：
$$
\begin{aligned}
    \frac{\partial L}{\partial F_{c_o, c_i, h_f, w_f}} 
    & = \sum_{b=1}^{B} \sum_{c'_o}^{C_{out}} \sum_{h_o}^{H_{out}} \sum_{w_o}^{W_{out}} \frac{\partial L}{\partial Y_{b,c'_o, h_o, w_o}} \frac{\partial Y_{b,c'_o, h_o, w_o}}{\partial F_{c_o, c_i, h_f, w_f}} \\
    & = \sum_{b=1}^{B} \sum_{h_o}^{H_{out}} \sum_{w_o}^{W_{out}} \frac{\partial L}{\partial Y_{b,c_o, h_o, w_o}} \frac{\partial Y_{b,c_o, h_o, w_o}}{\partial F_{c_o, c_i, h_f, w_f}}
\end{aligned}
$$

其中
$$
\begin{aligned}
    \frac{\partial Y_{b,c_o, h_o, w_o}}{\partial F_{c_o, c_i, h_f, w_f}}
    & = \frac{\partial }{\partial F_{c_o, c_i, h_f, w_f}} \left( \sum_{c'_i=1}^{C_{in}} \sum_{h'_f=1}^{H_f} \sum_{w'_i=1}^{W_{f}} F_{c_o, c'_i, h'_f, w'_f} X_{b, c'_i, s(h_o-1)+h'_f, s(w_o-1)+w'_f} \right) \\
    & = \frac{\partial }{\partial F_{c_o, c_i, h_f, w_f}} \left( F_{c_o, c_i, h_f, w_f} X_{b, c_i, s(h_o-1)+h_f, s(w_o-1)+w_f} \right) \\
    & = X_{b, c_i, s(h_o-1)+h_f, s(w_o-1)+w_f}
\end{aligned}
$$

故得
$$
\frac{\partial L}{\partial F_{c_o, c_i, h_f, w_f}} = \sum_{b=1}^{B} \sum_{h_o}^{H_{out}} \sum_{w_o}^{W_{out}} \frac{\partial L}{\partial Y_{b,c_o, h_o, w_o}}  X_{b, c_i, s(h_o-1)+h_f, s(w_o-1)+w_f}
$$

- 反向传播(误差返回到上一层)：
$$
\begin{aligned}
    \frac{\partial L}{\partial X_{b, c_i, h_i, w_i}} 
    & = \sum_{b'=1}^{B} \sum_{c_o}^{C_{out}} \sum_{h_o}^{H_{out}} \sum_{w_o}^{W_{out}} \frac{\partial L}{\partial Y_{b', c_o, h_o, w_o}} \frac{\partial Y_{b', c_o, h_o, w_o}}{\partial X_{b, c_i, h_i, w_i}} \\
    & = \sum_{c_o}^{C_{out}} \sum_{h_o}^{H_{out}} \sum_{w_o}^{W_{out}} \frac{\partial L}{\partial Y_{b, c_o, h_o, w_o}} \frac{\partial Y_{b, c_o, h_o, w_o}}{\partial X_{b, c_i, h_i, w_i}} \\
\end{aligned}
$$

其中
$$
\begin{aligned}
    \frac{\partial Y_{b, c_o, h_o, w_o}}{\partial X_{b, c_i, h_i, w_i}}
    &= \frac{\partial }{\partial X_{b, c_i, h_i, w_i}} \left( \sum_{c'_i=1}^{C_{in}} \sum_{h_f=1}^{H_f} \sum_{w_f=1}^{W_{f}} F_{c'_o, c_i, h_f, w_f} X_{b, c'_i, s(h_o-1)+h_f, s(w_o-1)+w_f} \right) \\
    &= \frac{\partial }{\partial X_{b, c_i, h_i, w_i}} \left( \sum_{h_f=1}^{H_f} \sum_{w_f=1}^{W_{f}} F_{c_o, c_i, h_f, w_f} X_{b, c_i, s(h_o-1)+h_f, s(w_o-1)+w_f} \right) \\
\end{aligned}
$$

仅对
$$
\begin{aligned}
    h_f = h_i - s(h_o - 1)\\
    w_f = w_i - s(w_o - 1)
\end{aligned}
$$
有偏导：
$$
\frac{\partial Y_{b, c_o, h_o, w_o}}{\partial X_{b, c_i, h_i, w_i}}
= F_{c_o, c_i, h_i - s(h_o - 1), w_i - s(w_o - 1)}
$$
且限制
$$
\max \left(1, \frac{h_i - H_f}{s} + 1\right) \leq h_o \leq \min \left( H_{out}, \frac{h_i - 1}{s} + 1 \right)\\
\max \left(1, \frac{w_i - W_f}{s} + 1\right) \leq w_o \leq \min \left( W_{out}, \frac{w_i - 1}{s} + 1 \right)
$$


故得
$$
\frac{\partial L}{\partial X_{b, c_i, h_i, w_i}} 
= \sum_{c_o}^{C_{out}} \sum_{h_o = \max \left(1, \frac{h_i - H_f}{s} + 1\right)}^{\min \left( H_{out}, \frac{h_i - 1}{s} + 1 \right)} \sum_{w_o = \max \left(1, \frac{w_i - W_f}{s} + 1\right)}^{\min \left( W_{out}, \frac{w_i - 1}{s} + 1 \right)} \frac{\partial L}{\partial Y_{b, c_o, h_o, w_o}} F_{c_o, c_i, h_i - s(h_o - 1), w_i - s(w_o - 1)}
$$

```c++
void ConvLayer::backward(Tensor4D &input, Tensor4D &output_grad, Tensor4D &input_grad){
    // 计算卷积核的梯度
    for(int c_out = 0; c_out < weight->size.len1; ++c_out){
        for(int c_in = 0; c_in < weight->size.len2; ++c_in){
            for(int h_f = 0; h_f < weight->size.len3; ++h_f){
                for(int w_f = 0; w_f < weight->size.len4; ++w_f){
                // ------------------------------------------------------------------------------
                    for(int batch = 0; batch < output_grad.size.len1; ++batch){
                        for(int h_out = 0; h_out < output_grad.size.len3; ++h_out){
                            for(int w_out = 0; w_out < output_grad.size.len4; ++w_out){
                                weight_grad->data[c_out][c_in][h_f][w_f] += output_grad.data[batch][c_out][h_out][w_out] * input.data[batch][c_in][stride*h_out+h_f][stride*w_out+w_f];
                            }
                        }
                    }
                // ------------------------------------------------------------------------------
                }
            }
        }
    }


    int h_f = weight->size.len3;
    int w_f = weight->size.len4;
    // 误差返回到上一层
    for(int batch = 0; batch < input_grad.size.len1; ++batch){
        for(int c_in = 0; c_in < input_grad.size.len2; ++c_in){
            for(int h_in = 0; h_in < input_grad.size.len3; ++h_in){
                    for(int w_in = 0; w_in < input_grad.size.len4; ++w_in){
                    // --------------------------------------------------------------------------
                        for(int c_out = 0; c_out < output_grad.size.len2; ++c_out){
                            for(int h_out = max((h_in - h_f) / stride + 1 ,0); h_out < min(h_in / stride + 1, output_grad.size.len3); ++h_out){
                                for(int w_out = max((w_in - w_f) / stride + 1, 0); w_out < min(w_in / stride + 1, output_grad.size.len4); ++w_out){
                                    input_grad.data[batch][c_in][h_in][w_in] += output_grad.data[batch][c_out][h_out][w_out] * weight->data[c_out][c_in][h_in - stride*h_out][w_in - stride*w_out];
                                }
                            }
                        }
                    // --------------------------------------------------------------------------
                }
            }
        }
    }

}
```

# ✅ MLP

## ▶ 前向传播

- 输入 $X \in \mathbb{R}^{B \times C_{in} \times 1 \times 1}$

- 参数为 $F \in \mathbb{R}^{C_{out} \times C_{in} \times  \times 1}$，步长为$s$

- 输出 $Y \in \mathbb{R}^{B \times C_{out} \times 1 \times 1}$

- 传播：
$$
Y_{b,c_o, 1, 1} = \sum_{c_i=1}^{C_{in}} F_{c_o, c_i, 1, 1} X_{b, c_i, 1, 1}
$$

```C++
void MLP::forward(Tensor4D &input, Tensor4D &output){
    for(int batch = 0; batch < output.size.len1; ++batch){
        for(int c_out = 0; c_out < output.size.len2; ++c_out){
            // output.data[batch][c_out][0][0] = 0.0;
            for(int c_in = 0; c_in < input.size.len2; ++c_in){
                output.data[batch][c_out][0][0] += input.data[batch][c_in][0][0] * weight->data[c_out][c_in][0][0];
            }
        }
    }
}
```

## ◀ 反向传播

- 输入$\frac{\partial L}{\partial Y} \in \mathbb{R}^{B \times C_{out} \times 1 \times 1}$，$X \in \mathbb{R}^{B \times C_{in} \times 1 \times 1}$， $F \in \mathbb{R}^{C_{out} \times C_{in} \times  \times 1}$

- 输出 $\frac{\partial L}{\partial X} \in \mathbb{R}^{B \times C_{in} \times 1 \times 1}$ 和 $\frac{\partial L}{\partial F} \in \mathbb{R}^{C_{out} \times C_{in} \times  \times 1}$

- 反向传播(对参数求偏导)：
  $$
  \begin{aligned}
      \frac{\partial L}{\partial F_{c_o, c_i, 1, 1}} 
      & = \sum_{b=1}^{B} \sum_{c'_o}^{C_{out}} \frac{\partial L}{\partial Y_{b,c'_o, 1, 1}} \frac{\partial Y_{b,c'_o, 1, 1}}{\partial F_{c_o, c_i, 1, 1}} \\
      & = \sum_{b=1}^{B}  \frac{\partial L}{\partial Y_{b,c_o, 1, 1}} \frac{\partial Y_{b, c_o, 1, 1}}{\partial F_{c_o, c_i, 1, 1}} \\
      & = \sum_{b=1}^{B} \frac{\partial L}{\partial Y_{b,c_o, 1, 1}} \frac{\partial }{\partial F_{c_o, c_i, 1, 1}} \left( \sum_{c'_i=1}^{C_{in}} F_{c_o, c'_i, 1, 1} X_{b, c'_i, 1, 1} \right)  \\
      & = \sum_{b=1}^{B} \frac{\partial L}{\partial Y_{b,c_o, 1, 1}} X_{b, c_i, 1, 1}
  \end{aligned}
  $$

- 反向传播(误差返回上一层)：
  $$
  \begin{aligned}
      \frac{\partial L}{\partial X_{b, c_i, 1, 1}} 
      & = \sum_{b'=1}^{B} \sum_{c_o}^{C_{out}} \frac{\partial L}{\partial Y_{b', c_o, 1, 1}} \frac{\partial Y_{b', c_o, 1, 1}}{\partial X_{b, c_i, 1, 1}} \\
      & = \sum_{c_o}^{C_{out}} \frac{\partial L}{\partial Y_{b, c_o, 1, 1}} \frac{\partial Y_{b, c_o, 1, 1}}{\partial X_{b, c_i, 1, 1}} \\
      & = \sum_{c_o}^{C_{out}} \frac{\partial L}{\partial Y_{b, c_o, 1, 1}} \frac{\partial }{\partial X_{b, c_i, 1, 1}} \left( \sum_{c'_i=1}^{C_{in}} F_{c_o, c'_i, 1, 1} X_{b, c'_i, 1, 1} \right)  \\
      & = \sum_{c_o}^{C_{out}} \frac{\partial L}{\partial Y_{b, c_o, 1, 1}} F_{c_o, c_i, 1, 1}\\
  \end{aligned}
  $$

```c++
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
```





# ✅ 池化层

## ▶ 前向传播

```c++
void MaxPool::forward(Tensor4D &input, Tensor4D &output){
    pool_map.clear();
    for(int batch = 0; batch < output.size.len1; ++batch){
        for(int c_out = 0; c_out < output.size.len2; ++c_out){
            for(int h_out = 0; h_out < output.size.len3; ++h_out){
                for(int w_out = 0; w_out < output.size.len4; ++w_out){
                    int res = -9999999;
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
```

## ◀ 反向传播

```c++
void MaxPool::backward(Tensor4D &output_grad, Tensor4D &input_grad){
    for(auto idx : pool_map){
        input_grad.data[idx.second.b][idx.second.c][idx.second.h][idx.second.w] = output_grad.data[idx.first.b][idx.first.c][idx.first.h][idx.first.w];
    }
}
```




# ✅ 展开层

## ▶ 前向传播

- 输入 $X \in \mathbb{R}^{B \times C_{in} \times H_{in} \times W_{in}}$

- 输出 $Y \in \mathbb{R}^{B \times (C_{in} \times H_{in} \times W_{in}) \times 1 \times 1}$


```c++
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
```

## ◀ 反向传播

```c++
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
```




# ✅ SoftMax激活层

## ▶ 前向传播

- 输入 $X \in \mathbb{R}^{B \times C_{in} \times 1 \times 1}$

- 输出 $Y \in \mathbb{R}^{B \times C_{out} \times 1 \times 1}$, $C_{out} = C_{in}$

- 传播：
$$
Y_{b, c_o, 1, 1} = \frac{e^{X_{b,c_o,1,1}}} {\sum_{c_i=1}^{C_{in}} e^{X_{b,c_i,1,1}}}
$$

```C++
void SoftMaxLayer::forward(Tensor4D &input, Tensor4D &output){ 
    op_exp<double> exp_fun;
    for(int batch = 0; batch < input.size.len1; ++batch){
        double max_num = -1e8;
        for(int c = 0; c < input.size.len2; ++c){
            max_num = max(max_num, input.data[batch][c][0][0]);
        }
        for(int c = 0; c < input.size.len2; ++c){
            output.data[batch][c][0][0] = exp_fun(input.data[batch][c][0][0] - max_num);
        }
        double sum = 0.0;
        for(int c = 0; c < output.size.len2; ++c){
            sum += output.data[batch][c][0][0];
        }
        for(int c = 0; c < output.size.len2; ++c){
            output.data[batch][c][0][0] /= sum;
        }
    }
};
```

## ◀ 反向传播

- 输入$\frac{\partial L}{\partial Y} \in \mathbb{R}^{B \times C_{out} \times 1 \times 1}$和 $Y \in \mathbb{R}^{B \times C_{out} \times 1 \times 1}$

- 输出 $\frac{\partial L}{\partial X} \in \mathbb{R}^{B \times C_{in} \times 1 \times 1}$ , $C_{out} = C_{in}$

- 反向传播：
  $$
  \begin{aligned}
  \frac{\partial L}{\partial X_{b, c_i, 1, 1}} 
  &= \sum_{b'=1}^{B}\sum_{c_o=1}^{C_{cout}} \frac{\partial L}{\partial Y_{b', c_o, 1, 1}} \frac{\partial Y_{b', c_o, 1, 1}}{\partial X_{b, c_i, 1, 1}} \\
  &= \sum_{c_o=1}^{C_{cout}} \frac{\partial L}{\partial Y_{b, c_o, 1, 1}} \frac{\partial Y_{b, c_o, 1, 1}}{\partial X_{b, c_i, 1, 1}} \\
  &= \sum_{c_o=1}^{C_{cout}} \frac{\partial L}{\partial Y_{b, c_o, 1, 1}} \frac{\partial}{\partial X_{b, c_i, 1, 1}} \left( \frac{e^{X_{b,c_o,1,1}}} {\sum_{c'_i=1}^{C_{in}} e^{X_{b,c'_i,1,1}}} \right) \\
  %&= \sum_{c_o=1}^{C_{cout}} \frac{\partial L}{\partial Y_{b, c_o, 1, 1}} \left( \frac{\frac{\partial e^{X_{b,c_o,1,1}}}{\partial X_{b, c_i, 1, 1}}(\sum_{c'_i=1}^{C_{in}} e^{X_{b,c'_i,1,1}}) -  e^{X_{b,c_o,1,1}} e^{X_{b,c_i,1,1}}} {(\sum_{c'_i=1}^{C_{in}} e^{X_{b,c'_i,1,1}})^2} \right) \\
  %&= \sum_{c_o=1}^{C_{cout}} \frac{\partial L}{\partial Y_{b, c_o, 1, 1}} \left( \frac{\frac{\partial e^{X_{b,c_o,1,1}}}{\partial X_{b, c_i, 1, 1}}}{\sum_{c'_i=1}^{C_{in}} e^{X_{b,c'_i,1,1}}} - \frac{e^{X_{b,c_o,1,1}}} {\sum_{c'_i=1}^{C_{in}} e^{X_{b,c'_i,1,1}}} \frac{e^{X_{b,c_i,1,1}}}{\sum_{c'_i=1}^{C_{in}} e^{X_{b,c'_i,1,1}}}\right) \\
  %&= \sum_{c_o=1}^{C_{cout}} \frac{\partial L}{\partial Y_{b, c_o, 1, 1}} \left( \frac{\frac{\partial e^{X_{b,c_o,1,1}}}{\partial X_{b, c_i, 1, 1}}}{\sum_{c'_i=1}^{C_{in}} e^{X_{b,c'_i,1,1}}} - Y_{b,c_o,1,1} Y_{b,c_i,1,1}\right) \\
  %&= - Y_{b,c_i,1,1} \sum_{c_o=1}^{C_{cout}} \frac{\partial L}{\partial Y_{b, c_o, 1, 1}} Y_{b,c_o,1,1} + \sum_{c_o=1}^{C_{cout}} \frac{\partial L}{\partial Y_{b, c_o, 1, 1}}\frac{\frac{\partial e^{X_{b,c_o,1,1}}}{\partial X_{b, c_i, 1, 1}}}{\sum_{c'_i=1}^{C_{in}} e^{X_{b,c'_i,1,1}}}\\
  \end{aligned}
  $$
  其中
  $$
  \begin{aligned}
  \frac{\partial}{\partial X_{b, c_i, 1, 1}} \left( \frac{e^{X_{b,c_o,1,1}}} {\sum_{c'_i=1}^{C_{in}} e^{X_{b,c'_i,1,1}}} \right)
  &= \frac{\frac{\partial e^{X_{b,c_o,1,1}}}{\partial X_{b, c_i, 1, 1}}(\sum_{c'_i=1}^{C_{in}} e^{X_{b,c'_i,1,1}}) -  e^{X_{b,c_o,1,1}} e^{X_{b,c_i,1,1}}} {(\sum_{c'_i=1}^{C_{in}} e^{X_{b,c'_i,1,1}})^2}\\
  &= \frac{\frac{\partial e^{X_{b,c_o,1,1}}}{\partial X_{b, c_i, 1, 1}}}{\sum_{c'_i=1}^{C_{in}} e^{X_{b,c'_i,1,1}}} - \frac{e^{X_{b,c_o,1,1}}} {\sum_{c'_i=1}^{C_{in}} e^{X_{b,c'_i,1,1}}} \frac{e^{X_{b,c_i,1,1}}}{\sum_{c'_i=1}^{C_{in}} e^{X_{b,c'_i,1,1}}} \\
  &= \frac{\frac{\partial e^{X_{b,c_o,1,1}}}{\partial X_{b, c_i, 1, 1}}}{\sum_{c'_i=1}^{C_{in}} e^{X_{b,c'_i,1,1}}} - Y_{b,c_o,1,1} Y_{b,c_i,1,1} \\
  &= - Y_{b,c_o,1,1} Y_{b,c_i,1,1} + \begin{cases}
  Y_{b,c_o,1,1} , \quad c_o = c_i,\\
  0, \quad c_o \neq c_i
  \end{cases}
  \end{aligned}
  $$
  

  ```c++
  void SoftMaxLayer::backward(Tensor4D &output, Tensor4D &output_grad, Tensor4D &input_grad){ 
      for(int batch = 0; batch < input_grad.size.len1; ++batch){
          for(int c_in = 0; c_in < input_grad.size.len2; ++c_in){
              for(int c_out = 0; c_out < output_grad.size.len2; ++c_out){
                  input_grad.data[batch][c_in][0][0] += - output_grad.data[batch][c_out][0][0] * output.data[batch][c_out][0][0] * output.data[batch][c_in][0][0];
                  if(c_out == c_in) input_grad.data[batch][c_in][0][0] += output_grad.data[batch][c_out][0][0] * output.data[batch][c_in][0][0];
              }
          }
      }
  };
  ```

  


# ✅ CrossEntropy损失层

## ▶ 前向传播

- 输入 $X \in \mathbb{R}^{B \times C_{in} \times 1 \times 1}$ 和 $L \in \mathbb{R}^{B \times C_{in} \times 1 \times 1}$
- 输出 $Loss \in \mathbb{R}^{1 \times 1 \times 1 \times 1}$
- 传播：
$$
Loss = - \frac{1}{B}\sum_{b}^{B} \sum_{c_i}^{C_{in}} L_{b, c_i, 1, 1} \log X_{b, c_i, 1, 1}
$$

```c++
void CrossEntropyLossLayer::forward(Tensor4D &input, Tensor4D &label){
    _loss = 0.0;
    for(int batch = 0; batch < input.size.len1; ++batch){
        for(int c = 0; c < input.size.len2; ++c){
            _loss += - label.data[batch][c][0][0] * log(input.data[batch][c][0][0]);
        }
    }
```

## ◀ 反向传播

- 输入$Loss \in \mathbb{R}^{1 \times 1 \times 1 \times 1}$，$X \in \mathbb{R}^{B \times C_{in} \times 1 \times 1}$ 和$L \in \mathbb{R}^{B \times C_{in} \times 1 \times 1}$

- 输出$\frac{\partial Loss}{\partial X} \in \mathbb{R}^{B \times C_{in} \times 1 \times 1}$

- 反向传播：
$$
\begin{aligned}
\frac{\partial Loss}{\partial X_{b, c_i, 1, 1}} 
& = - \frac{1}{B} \frac{\partial}{\partial X_{b, c_i, 1, 1}}  \left(\sum_{b'}^{B} \sum_{c'_i}^{C_{in}} L_{b', c'_i, 1, 1} \log X_{b', c'_i, 1, 1} \right) \\
& = - \frac{1}{B} \frac{\partial}{\partial X_{b, c_i, 1, 1}}  \left( L_{b, c_i, 1, 1} \log X_{b, c_i, 1, 1} \right) \\
& = - \frac{1}{B} \frac{L_{b, c_i, 1, 1}}{X_{b, c_i, 1, 1}} \\
\end{aligned}
$$

```c++
void CrossEntropyLossLayer::backward(Tensor4D &input, Tensor4D &label, Tensor4D &input_grad){
    double B = 1.0 / input.size.len1;
    for(int batch = 0; batch < input_grad.size.len1; ++batch)
        for(int c = 0; c < input_grad.size.len2; ++c)
            input_grad.data[batch][c][0][0] = - B * label.data[batch][c][0][0] / input.data[batch][c][0][0];
}
```

## ◀ 反向传播（配合softmax）

$$
\begin{aligned}
Y_{b, c_o, 1, 1} &= \frac{e^{X_{b,c_o,1,1}}} {\sum_{c_i=1}^{C_{in}} e^{X_{b,c_i,1,1}}} \\
Loss &= - \frac{1}{B}\sum_{b}^{B} \sum_{c_o}^{C} L_{b, c_o, 1, 1} \log Y_{b, c_o, 1, 1}
\end{aligned}
$$

那么反向传播则为
$$
\begin{aligned}
\frac{\partial Loss}{\partial X_{b,c_i,1,1}}
&= \sum_{c_o=1}^{C} \frac{\partial Loss}{\partial Y_{b,c_o,1,1}} \frac{\partial Y_{b,c_o,1,1}}{\partial X_{b,c_i,1,1}} \\
&= - \frac{1}{B} \sum_{c_o=1}^{C} \frac{L_{b, c_o, 1, 1}}{Y_{b,c_o,1,1}} \frac{\partial Y_{b,c_o,1,1}}{\partial X_{b,c_i,1,1}} \\
&= - \frac{1}{B} \sum_{c_o=1}^{C} \frac{L_{b, c_o, 1, 1}}{Y_{b,c_o,1,1}} \left( - Y_{b,c_o,1,1} Y_{b,c_i,1,1} + \mathbb{1}(c_i = c_o) Y_{b,c_o,1,1} \right) \\
&= - \frac{1}{B} \sum_{c_o=1}^{C} \left( - L_{b, c_o, 1, 1} Y_{b,c_i,1,1} + \mathbb{1}(c_i = c_o) L_{b, c_o, 1, 1} \right) \\
&=\frac{1}{B} \left[ Y_{b,c_i,1,1} ( \sum_{c_o=1}^{C} L_{b, c_o, 1, 1} ) - L_{b, c_i, 1, 1} \right] \\
&=\frac{Y_{b,c_i,1,1} - L_{b, c_i, 1, 1}}{B} \\
\end{aligned}
$$


```c++
void CrossEntropyLossLayer::backward_skip(Tensor4D &softmax_out, Tensor4D &label, Tensor4D &softmax_input_grad){
    double B = 1.0 / softmax_input_grad.size.len1;
    for(int batch = 0; batch < softmax_input_grad.size.len1; ++batch)
        for(int c = 0; c < softmax_input_grad.size.len2; ++c)
            softmax_input_grad.data[batch][c][0][0] = B * (softmax_out.data[batch][c][0][0] - label.data[batch][c][0][0]);
}
```

# ✅ BatchNorm层

## ▶ 前向传播

- 输入 $X \in \mathbb{R}^{B \times C \times H \times W}$

- 中间参数为 $\mu \in \mathbb{R}^{C_{in} \times 1 \times  \times 1}$ 和 $\sigma^2 \in \mathbb{R}^{C_{in} \times 1 \times \times 1}$

- 学习参数$\gamma,\beta$

- 输出 $Y \in \mathbb{R}^{B \times C \times H \times W}$

- 传播：
  $$
  \begin{aligned}
  \mu_{c} &= \frac{1}{BHW} \sum_{b}^{B} \sum_{h}^{H} \sum_{w}^{W} X_{b,c,h,w} \\
  \sigma_{c}^2 &= \frac{1}{BHW} \sum_{b}^{B} \sum_{h}^{H} \sum_{w}^{W} \left( X_{b,c,h,w} - \mu_c \right)^2 \\
  \hat{X}_{b,c,h,w} &= \frac{X_{b,c,h,w} - \mu_c}{\sqrt{\sigma_c^2 + \varepsilon}} \\
  Y_{b,c,h,w} &= \gamma_{c} \hat{X}_{b,c,h,w} + \beta_{c}
  \end{aligned}
  $$

```C++
void BatchNormLayer::forward(Tensor4D &input, Tensor4D &output){
    double BHW = 1.0 / input.size.len1 * input.size.len3 * input.size.len4;
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
                    output.data[b][c][h][w] = gamma * (input.data[b][c][h][w] - Mean->data[c][0][0][0]) / pow(Var->data[c][0][0][0] + 1e-6, 0.5) + beta;
    
    input_data = &input;
};
```

## ◀ 反向传播


$$
\begin{aligned}
\frac{\partial Loss}{\partial \hat{X}_{b,c,h,w}} &= \gamma_c \frac{\partial Loss}{\partial Y_{b,c,h,w}} \\
\frac{\partial Loss}{\partial \sigma_c^2} &= - \frac{1}{2 \left( \sigma_c^2 + \varepsilon\right)^{3/2}}\sum_{b}^{B} \sum_{h}^{H} \sum_{w}^{W} \frac{\partial Loss}{\partial \hat{X}_{b,c,h,w}} \left( X_{b,c,h,w} - \mu_c \right) \\
\frac{\partial Loss}{\partial \mu_c} &= - \frac{1}{\left( \sigma_c^2 + \varepsilon\right)^{1/2}} \sum_{b}^{B} \sum_{h}^{H} \sum_{w}^{W} \frac{\partial Loss}{\partial \hat{X}_{b,c,h,w}}
+ \frac{\partial Loss}{\partial \sigma_c^2} \frac{-2\sum_{b}^{B} \sum_{h}^{H} \sum_{w}^{W} (X_{b,c,h,w} - \mu_c)}{BHW} \\
\frac{\partial Loss}{\partial X_{b,c,h,w}} &= \frac{1}{\left( \sigma_c^2 + \varepsilon\right)^{1/2}} \frac{\partial Loss}{\partial \hat{X}_{b,c,h,w}} + \frac{2(X_{b,c,h,w} - \mu_c)}{BHW} \frac{\partial Loss}{\partial \sigma_c^2} + \frac{1}{BHW} \frac{\partial Loss}{\partial \mu_c}\\
\frac{\partial Loss}{\partial \gamma_c} &= \sum_{b}^{B} \sum_{h}^{H} \sum_{w}^{W} \frac{\partial Loss}{\partial Y_{b,c,h,w}} \hat{X}_{b,c,h,w} \\
\frac{\partial Loss}{\partial \beta_c} &= \sum_{b}^{B} \sum_{h}^{H} \sum_{w}^{W} \frac{\partial Loss}{\partial Y_{b,c,h,w}}
\end{aligned}
$$

```C++
void BatchNormLayer::backward(Tensor4D &output, Tensor4D &output_grad, Tensor4D &input_grad){ 
    for(int c = 0; c < input_grad.size.len2; ++c){
        gamma_grad->data[c][0][0][0] = 0.0;
        beta_grad->data[c][0][0][0] = 0.0;
    }
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
```

