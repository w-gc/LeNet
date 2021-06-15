#include<iostream>
#include"getIMG.cpp"
#include"tensor.cpp"
#include"LeNet.cpp"
using namespace std;

void printLabel(Tensor4D &label){
	cout << "label : [";
	for(int batch = 0; batch < label.size.len1; ++batch){
        double max_num = label.data[batch][0][0][0];
		int idx = 0;
        for(int c = 0; c < label.size.len2; ++c){
            if( max_num < label.data[batch][c][0][0] ){
				max_num = label.data[batch][c][0][0];
				idx = c;
			}
        }
		cout << idx << ", ";
	}
	cout << "]" << endl;
}

void printDigit(Tensor4D &IMG){
	for(int i = 0; i < IMG.size.len1; ++i){
		cout << "[";
        for(int j = 0; j < IMG.size.len2; ++j){
			cout << "[";
            for(int k = 0; k < IMG.size.len3; ++k) {
                cout << "[";
                for(int l = 0; l < IMG.size.len4; ++l) {
                    // cout << setprecision(2) << data[i][j][k][l] << ", ";
					if(IMG.data[i][j][k][l] <= 0.5) cout << "  ";
					else cout << "* ";
                }
                cout << "]," << endl;
            }
			cout << "]," << endl;
		}
		cout << "]" << endl;
	}
}

int main(){
	int batch_size = 30;
	int MAX_EPOCH = 1;
	double loss = 0.0;
	double acc = 0.0;
	double norm_fac = 1.0/255.0;
	double lr = 0.02;

	getBatchImage *bimage = new getBatchImage("../data_set/train-images.idx3-ubyte", "../data_set/train-labels.idx1-ubyte", batch_size);
	bimage->batchImage[0].printSize();
	bimage->batchLabel[0].printSize();
	

	norm_fac = bimage->batchImage[0].Max();
	LeNet lenet(TensorShape{batch_size, 1, 28, 28}, TensorShape{batch_size, 10, 1, 1}, lr);
	for(int bn = 0; bn < bimage->batch_num; ++bn){
		bimage->batchImage[bn].scalarOpTensor_(norm_fac, op_mul<double>());
	}

	Tensor4D *input;
	Tensor4D *label;
	
	for(int iter = 0; iter < MAX_EPOCH; ++iter){
		cout << "------------------- EPOCH: " << iter << "------------------" << endl; 
		for(int bn = 0; bn < bimage->batch_num; ++bn){

			input = &(bimage->batchImage[bn]);
			label = &(bimage->batchLabel[bn]);
			cout << "bn = " << bn << ",  "; 

			lenet.layer_zero();
			lenet.forward(*input);

			loss = lenet.loss(*label);
			acc = lenet.accuracy(*label);
			cout << "loss : " << loss << ", acc: " << acc << endl;

			// lenet.prediction();
			// printLabel(*label);

			lenet.grad_zero();
			lenet.backward(*label);

			lenet.update();
		}
	}
	
	return 0;
}

// g++ run.cpp -o run
// ./run 1>./1>train.log 2>&1 &
// nohup ./run 1>run.log 2>&1 &