#include"time.h"
#include<cmath>
#include<ctime>
#include<fstream>
#include<string>
#include<cstdlib>
#include<iostream> 
#include"tensor.cpp"
using namespace std;

// #define IMAGE_HEIGHT 28
// #define IMAGE_WIDTH 28
#define OUTPUT_SIZE 10


class getBatchImage{
public:
	Tensor4D* batchImage;
    Tensor4D* batchLabel;
	int batch_num;
	int batch_size;
	
	getBatchImage(const char *datapath, const char *labelpath, int batch_size){
		this->batch_size = batch_size;
		ifstream file(datapath, ios::binary);
		if (file.is_open()){
			int magic_number = 0;
			int number_of_images = 0;
			int n_rows = 0;
			int n_cols = 0;
			unsigned char label;
			file.read((char*)&magic_number, sizeof(magic_number));
			file.read((char*)&number_of_images, sizeof(number_of_images));
			file.read((char*)&n_rows, sizeof(n_rows));
			file.read((char*)&n_cols, sizeof(n_cols));
					
			magic_number = ((int)(magic_number & 255) << 24) + ((int)((magic_number >> 8) & 255) << 16) 
							+ ((int)((magic_number >> 16) & 255) << 8) + ((magic_number >> 24) & 255);
			number_of_images = ((int)(number_of_images & 255) << 24) + ((int)((number_of_images >> 8) & 255) << 16) 
							+ ((int)((number_of_images >> 16) & 255) << 8) + ((number_of_images >> 24) & 255);
			n_rows = ((int)(n_rows & 255) << 24) + ((int)((n_rows >> 8) & 255) << 16) 
							+ ((int)((n_rows >> 16) & 255) << 8) + ((n_rows >> 24) & 255);
			n_cols = ((int)(n_cols & 255) << 24) + ((int)((n_cols >> 8) & 255) << 16) 
							+ ((int)((n_cols >> 16) & 255) << 8) + ((n_cols >> 24) & 255);

			this->batch_num = number_of_images / this->batch_size;
			assert(number_of_images % this->batch_size == 0);
 			cout << "number of batch = " << this->batch_num <<endl;
			cout << "magic number = " << magic_number << endl;
			cout << "number of images = " << number_of_images << endl;
			cout << "rows = " << n_rows << endl;
			cout << "cols = " << n_cols << endl;

 			batchImage = new Tensor4D[this->batch_num];
			for (int i = 0; i < this->batch_num; i++) batchImage[i].NewATensor(TensorShape{this->batch_size, 1, 28, 28});

			batchLabel = new Tensor4D[this->batch_num];
			for (int i = 0; i < this->batch_num; i++) batchLabel[i].NewATensor(TensorShape{this->batch_size, OUTPUT_SIZE, 1, 1});
 			
			for (int i = 0; i < number_of_images; i++){
				int b1 = i / this->batch_size;
				int b2 = i % this->batch_size;
				for (int r = 0; r < n_rows; r++){
					for (int c = 0; c < n_cols; c++){
						unsigned char image = 0;
						file.read((char*)&image, sizeof(image));
						batchImage[b1].data[b2][0][r][c] = double(image) / 255.0;
					}
				}
			}
			
		}
		file.close();
		ifstream label_file(labelpath, ios::binary);
		if (label_file.is_open()){
			int magic_number = 0;
			int number_of_images = 0;
			label_file.read((char*)&magic_number, sizeof(magic_number));
			label_file.read((char*)&number_of_images, sizeof(number_of_images));
			magic_number = ((int)(magic_number & 255) << 24) + ((int)((magic_number >> 8) & 255) << 16) 
							+ ((int)((magic_number >> 16) & 255) << 8) + ((magic_number >> 24) & 255);
			number_of_images = ((int)(number_of_images & 255) << 24) + ((int)((number_of_images >> 8) & 255) << 16) 
							+ ((int)((number_of_images >> 16) & 255) << 8) + ((number_of_images >> 24) & 255);
			cout << "magic number = " << magic_number << endl;
			cout << "number of images = " << number_of_images << endl;
			for (int i = 0; i < number_of_images; i++){
				int b1 = i / this->batch_size;
				int b2 = i % this->batch_size;
				unsigned char label;
				label_file.read((char*)&label, sizeof(label));
				for(int j = 0; j < OUTPUT_SIZE; j++) batchLabel[b1].data[b2][j][0][0] = 0.0;				
				batchLabel[b1].data[b2][int(label)][0][0] += 1.0;
			}
		}
		label_file.close();
	}
	~getBatchImage(){ 
		delete []batchImage;
		delete []batchLabel;
	}
}; 
