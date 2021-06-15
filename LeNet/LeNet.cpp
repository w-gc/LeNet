#include<iostream>
#include<cmath>
#include<ctime>
#include<cstdlib>
#include"tensor.cpp"
#include"ConvLayer.cpp"
#include"PoolLayer.cpp"
#include"FlattenLayer.cpp"
#include"MLP.cpp"
#include"ActiLayer.cpp"
#include"LossLayer.cpp"
#include"Optimizer.cpp"
#include"NormLayer.cpp"
using namespace std;

class LeNet{
public:
	LeNet(TensorShape in_size, TensorShape out_size, double lr);
	void forward(Tensor4D &input);
	double loss(Tensor4D &label);
	void backward(Tensor4D &label);
	void update();
	void layer_zero();
	void grad_zero();
	void prediction();
	double accuracy(Tensor4D &label);
private:
	int batch_size;
	double _loss;
	double _acc;
	TensorShape input_size;
	TensorShape output_size;
	Optimizer *sgd;
	
	ConvLayer *conv1;
	BatchNormLayer *normlayer1;
	PoolLayer *pool1;

	ConvLayer *conv2;
	BatchNormLayer *normlayer2;
	PoolLayer *pool2;

	FlattenLayer *flatten;

	MLP *mlp1;
	BatchNormLayer *normlayer3;

	MLP *mlp2;
	BatchNormLayer *normlayer4;

	MLP *mlp3;
	SoftMaxLayer *softmax;

	ActiLayer *act;
	LossLayer *losslayer;

	Tensor4D *input_data;

	Tensor4D *conv1_out;
	Tensor4D *norm1_out;
	Tensor4D *act1_out;
	Tensor4D *pool1_out;

	Tensor4D *conv2_out;
	Tensor4D *norm2_out;
	Tensor4D *act2_out;
	Tensor4D *pool2_out;

	Tensor4D *flatten_out;

	Tensor4D *mlp1_out;
	Tensor4D *norm3_out;
	Tensor4D *act3_out;

	Tensor4D *mlp2_out;
	Tensor4D *norm4_out;
	Tensor4D *act4_out;

	Tensor4D *mlp3_out;

	Tensor4D *softmax_out;

	Tensor4D *softmax_out_grad;
	Tensor4D *mlp3_out_grad;

	Tensor4D *act4_out_grad;
	Tensor4D *norm4_out_grad;
	Tensor4D *mlp2_out_grad;

	Tensor4D *act3_out_grad;
	Tensor4D *norm3_out_grad;
	Tensor4D *mlp1_out_grad;

	Tensor4D *flatten_out_grad;

	Tensor4D *pool2_out_grad;
	Tensor4D *act2_out_grad;
	Tensor4D *norm2_out_grad;
	Tensor4D *conv2_out_grad;

	Tensor4D *pool1_out_grad;
	Tensor4D *act1_out_grad;
	Tensor4D *norm1_out_grad;
	Tensor4D *conv1_out_grad;

	Tensor4D *input_grad;
};


LeNet::LeNet(TensorShape in_size, TensorShape out_size, double lr): input_size(in_size), output_size(out_size), sgd(new SGD(lr, 1e-3)), _loss(0.0f), _acc(0.0f){
	batch_size = in_size.len1;

	conv1 = new ConvLayer(TensorShape{6, 1, 5, 5}, 1, sgd);
	normlayer1 = new BatchNormLayer(6, sgd);
	pool1 = new AvgPool(2, 2, 2); // new MaxPool(2, 2, 2); // 

	conv2 = new ConvLayer(TensorShape{16, 6, 5, 5}, 1, sgd);
	normlayer2 = new BatchNormLayer(16, sgd);
	pool2 = new AvgPool(2, 2, 2); // new MaxPool(2, 2, 2); // 

	flatten = new FlattenLayer();

	mlp1 = new MLP(TensorShape{120, 256, 1, 1}, sgd);
	normlayer3 = new BatchNormLayer(120, sgd);

	mlp2 = new MLP(TensorShape{80, 120, 1, 1}, sgd);
	normlayer4 = new BatchNormLayer(80, sgd);

	mlp3 = new MLP(TensorShape{10, 80, 1, 1}, sgd);
	softmax = new SoftMaxLayer();
	act = new ReLULayer(); // new SigmoidLayer();   // 
	losslayer = new CrossEntropyLossLayer(); // new QuadLossLayer(); // 


	conv1_out = new Tensor4D(TensorShape{batch_size, 6, 24, 24});
	norm1_out = new Tensor4D(TensorShape{batch_size, 6, 24, 24});
	act1_out = new Tensor4D(TensorShape{batch_size, 6, 24, 24});
	pool1_out = new Tensor4D(TensorShape{batch_size, 6, 12, 12});

	conv2_out = new Tensor4D(TensorShape{batch_size, 16, 8, 8});
	norm2_out = new Tensor4D(TensorShape{batch_size, 16, 8, 8});
	act2_out = new Tensor4D(TensorShape{batch_size, 16, 8, 8});
	pool2_out = new Tensor4D(TensorShape{batch_size, 16, 4, 4});

	flatten_out = new Tensor4D(TensorShape{batch_size, 256, 1, 1});

	mlp1_out = new Tensor4D(TensorShape{batch_size, 120, 1, 1});
	norm3_out = new Tensor4D(TensorShape{batch_size, 120, 1, 1});
	act3_out = new Tensor4D(TensorShape{batch_size, 120, 1, 1});

	mlp2_out = new Tensor4D(TensorShape{batch_size, 80, 1, 1});
	norm4_out = new Tensor4D(TensorShape{batch_size, 80, 1, 1});
	act4_out = new Tensor4D(TensorShape{batch_size, 80, 1, 1});

	mlp3_out = new Tensor4D(TensorShape{batch_size, 10, 1, 1});

	softmax_out = new Tensor4D(TensorShape{batch_size, 10, 1, 1});


	// ------------------------------------------------------------------

	softmax_out_grad = new Tensor4D(TensorShape{batch_size, 10, 1, 1});

	mlp3_out_grad = new Tensor4D(TensorShape{batch_size, 10, 1, 1});

	act4_out_grad = new Tensor4D(TensorShape{batch_size, 80, 1, 1});
	norm4_out_grad = new Tensor4D(TensorShape{batch_size, 80, 1, 1});
	mlp2_out_grad = new Tensor4D(TensorShape{batch_size, 80, 1, 1});

	act3_out_grad = new Tensor4D(TensorShape{batch_size, 120, 1, 1});
	norm3_out_grad = new Tensor4D(TensorShape{batch_size, 120, 1, 1});
	mlp1_out_grad = new Tensor4D(TensorShape{batch_size, 120, 1, 1});

	flatten_out_grad = new Tensor4D(TensorShape{batch_size, 256, 1, 1});

	pool2_out_grad = new Tensor4D(TensorShape{batch_size, 16, 4, 4});
	act2_out_grad = new Tensor4D(TensorShape{batch_size, 16, 8, 8});
	norm2_out_grad = new Tensor4D(TensorShape{batch_size, 16, 8, 8});
	conv2_out_grad = new Tensor4D(TensorShape{batch_size, 16, 8, 8});

	pool1_out_grad = new Tensor4D(TensorShape{batch_size, 6, 12, 12});
	act1_out_grad = new Tensor4D(TensorShape{batch_size, 6, 24, 24});
	norm1_out_grad = new Tensor4D(TensorShape{batch_size, 6, 24, 24});
	conv1_out_grad = new Tensor4D(TensorShape{batch_size, 6, 24, 24});

	input_grad = new Tensor4D(in_size);
	// --------------- load the pre-train weights
	// conv1->weight->loadData("../weights/conv1_weight.txt");
	// conv2->weight->loadData("../weights/conv2_weight.txt");
	// mlp1->weight->loadData("../weights/mlp1_weight.txt");
	// mlp2->weight->loadData("../weights/mlp2_weight.txt");
	// mlp3->weight->loadData("../weights/mlp3_weight.txt");
};

void LeNet::forward(Tensor4D &input){
	input_data = &input;

	conv1->forward(input, *conv1_out);
	normlayer1->forward(*conv1_out, *norm1_out);
	act->forward(*norm1_out, *act1_out);
	pool1->forward(*act1_out, *pool1_out);

	conv2->forward(*pool1_out, *conv2_out);
	normlayer2->forward(*conv2_out, *norm2_out);
	act->forward(*norm2_out, *act2_out);
	pool2->forward(*act2_out, *pool2_out);

	flatten->forward(*pool2_out, *flatten_out);

	mlp1->forward(*flatten_out, *mlp1_out);
	normlayer3->forward(*mlp1_out, *norm3_out);
	act->forward(*norm3_out, *act3_out);

	mlp2->forward(*act3_out, *mlp2_out);
	normlayer4->forward(*mlp2_out, *norm4_out);
	act->forward(*norm4_out, *act4_out);

	mlp3->forward(*act4_out, *mlp3_out);

	softmax->forward(*mlp3_out, *softmax_out);
}

double LeNet::loss(Tensor4D &label){
	losslayer->forward(*softmax_out, label);
	_loss = losslayer->loss();
	return _loss;
}

void LeNet::backward(Tensor4D &label){
	dynamic_cast<CrossEntropyLossLayer*>(losslayer)->backward_skip(*softmax_out, label, *mlp3_out_grad);
	// losslayer->backward(*softmax_out, label, *softmax_out_grad);
	// softmax->backward(*softmax_out, *softmax_out_grad, *mlp3_out_grad);

	mlp3->backward(*act4_out, *mlp3_out_grad, *act4_out_grad);

	act->backward(*norm4_out, *act4_out_grad, *norm4_out_grad);
	normlayer4->backward(*mlp2_out, *norm4_out_grad, *mlp2_out_grad);
	mlp2->backward(*act3_out, *mlp2_out_grad, *act3_out_grad);

	act->backward(*norm3_out, *act3_out_grad, *norm3_out_grad);
	normlayer3->backward(*mlp1_out, *norm3_out_grad, *mlp1_out_grad);
	mlp1->backward(*flatten_out, *mlp1_out_grad, *flatten_out_grad);

	flatten->backward(*flatten_out_grad, *pool2_out_grad);

	pool2->backward(*pool2_out_grad, *act2_out_grad);
	act->backward(*norm2_out, *act2_out_grad, *norm2_out_grad);
	normlayer2->backward(*conv2_out, *norm2_out_grad, *conv2_out_grad);
	conv2->backward(*pool1_out, *conv2_out_grad, *pool1_out_grad);

	pool1->backward(*pool1_out_grad, *act1_out_grad);
	act->backward(*norm1_out, *act1_out_grad, *norm1_out_grad);
	normlayer1->backward(*conv1_out, *norm1_out_grad, *conv1_out_grad);
	conv1->backward(*input_data, *conv1_out_grad, *input_grad);
}

void LeNet::update(){
	mlp3->update();

	// normlayer4->update();
	mlp2->update();

	// normlayer3->update();
	mlp1->update();

	// normlayer2->update();
	conv2->update();

	// normlayer1->update();
	conv1->update();
}

void LeNet::layer_zero(){
	conv1_out->init_zero();
	norm1_out->init_zero();
	act1_out->init_zero();
	pool1_out->init_zero();

	conv2_out->init_zero();
	norm2_out->init_zero();
	act2_out->init_zero();
	pool2_out->init_zero();

	flatten_out->init_zero();

	mlp1_out->init_zero();
	norm3_out->init_zero();
	act3_out->init_zero();

	mlp2_out->init_zero();
	norm4_out->init_zero();
	act4_out->init_zero();

	mlp3_out->init_zero();
	softmax_out->init_zero();

	normlayer1->layer_zero();
	normlayer2->layer_zero();
	normlayer3->layer_zero();
	normlayer4->layer_zero();

	
	softmax_out_grad->init_zero();
	mlp3_out_grad->init_zero();
	act4_out_grad->init_zero();
	norm4_out_grad->init_zero();
	mlp2_out_grad->init_zero();
	act3_out_grad->init_zero();
	norm3_out_grad->init_zero();
	mlp1_out_grad->init_zero();
	flatten_out_grad->init_zero();
	pool2_out_grad->init_zero();
	act2_out_grad->init_zero();
	norm2_out_grad->init_zero();
	conv2_out_grad->init_zero();
	pool1_out_grad->init_zero();
	act1_out_grad->init_zero();
	norm1_out_grad->init_zero();
	conv1_out_grad->init_zero();
}

void LeNet::grad_zero(){
	mlp3->grad_zero();
	mlp2->grad_zero();
	mlp1->grad_zero();
	conv2->grad_zero();
	conv1->grad_zero();

	normlayer1->grad_zero();
	normlayer2->grad_zero();
	normlayer3->grad_zero();
	normlayer4->grad_zero();
}

void LeNet::prediction(){
	cout << "pred : [";
	for(int batch = 0; batch < softmax_out->size.len1; ++batch){
        double max_num = softmax_out->data[batch][0][0][0];
		int idx = 0;
        for(int c = 0; c < softmax_out->size.len2; ++c){
            if( max_num < softmax_out->data[batch][c][0][0] ){
				max_num = softmax_out->data[batch][c][0][0];
				idx = c;
			}
        }
		cout << idx << ", ";
	}
	cout << "]" << endl;
}

double LeNet::accuracy(Tensor4D &label){
	int cnt = 0;
	for(int batch = 0; batch < softmax_out->size.len1; ++batch){
        double max_num = softmax_out->data[batch][0][0][0];
		int idx = 0;
        for(int c = 0; c < softmax_out->size.len2; ++c){
            if( max_num < softmax_out->data[batch][c][0][0] ){
				max_num = softmax_out->data[batch][c][0][0];
				idx = c;
			}
        }
		if(label.data[batch][idx][0][0] > 0.9) ++cnt;
	}
	_acc = cnt / double(softmax_out->size.len1);
	return _acc;
}