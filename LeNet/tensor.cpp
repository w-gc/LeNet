#ifndef TENSOR_CPP
#define TENSOR_CPP

#include"utilize.cpp"
#include<iostream>  
#include<iomanip>
#include<cassert>

#include<random> 
#include<iostream>
#include<fstream>

using namespace std;

#define EPSILON 1e-16

struct TensorShape{
    int len1, len2, len3, len4;
    void print(){ cout << "size: (" << len1 << "," << len2 << "," << len3 << "," << len4 << ")" << endl;}
    friend bool operator==(const TensorShape &a, const TensorShape &b){
        return a.len1 == b.len1 && a.len2 == b.len2 && a.len3 == b.len3 && a.len4 == b.len4;
    }
};


class Tensor4D{
public:
    TensorShape size;
    double ****data;
    Tensor4D(){};
    Tensor4D(TensorShape size);
    Tensor4D(const Tensor4D& Other);
    ~Tensor4D();
    void init_zero();
    void init_rand();
	void printData();
	void printSize();
    double Max();
    double Min();
    void printMaxMin();
    void cutoff();

    void saveData(string filename);
    void loadData(string filename);

    void NewATensor(TensorShape size);
    
    Tensor4D& operator=(const Tensor4D& Other);

    template<typename T, typename Op> 
    Tensor4D& scalarOpTensor_(T &a, Op op);

    template<typename Op>
    Tensor4D& TensorOpTensor_(const Tensor4D &Other, Op op);

    template<typename T, typename Op>
    void scalarOpTensor(T &a, Tensor4D &output, Op op);

    template<typename Op>
    void TensorOpTensor(const Tensor4D &Other, Tensor4D &output, Op op);

    template<typename Op> 
    Tensor4D& pointwiseFun_(Op op);

    template<typename Op> 
    void pointwiseFun(const Tensor4D &output, Op op);
}; 

Tensor4D::Tensor4D(TensorShape size){ NewATensor(size); }

Tensor4D::Tensor4D(const Tensor4D& Other){
    size = Other.size;
    NewATensor(size);
    *this = Other;
}

Tensor4D::~Tensor4D(){
    for(int i = 0; i < size.len1; i++){
        for(int j = 0; j < size.len2; j++){
            for(int k = 0; k < size.len3; ++k) delete []data[i][j][k];
            delete []data[i][j];
        } 
        delete []data[i];
    }
    delete []data;
}

void Tensor4D::init_zero(){
    for(int i = 0; i < size.len1; i++)
        for(int j = 0; j < size.len2; j++)
            for(int k = 0; k < size.len3; k++)
                for(int l = 0; l < size.len4; ++l)
                    data[i][j][k][l] = 0.0;
}

void Tensor4D::init_rand(){
    std::default_random_engine random(time(NULL));  
    std::normal_distribution<double> dis(0.0, 1.0);  
    for(int i = 0; i < size.len1; i++)
        for(int j = 0; j < size.len2; j++)
            for(int k = 0; k < size.len3; k++)
                for(int l = 0; l < size.len4; ++l)
                    data[i][j][k][l] = dis(random);
}

void Tensor4D::printData(){
	for(int i = 0; i < size.len1; ++i){
		cout << "[";
        for(int j = 0; j < size.len2; ++j){
			cout << "[";
            for(int k = 0; k < size.len3; ++k) {
                cout << "[";
                for(int l = 0; l < size.len4; ++l) {
                    cout << setprecision(2) << data[i][j][k][l] << ", ";
                }
                cout << "]," << endl;
            }
			cout << "]," << endl;
		}
		cout << "]" << endl;
	}
}

void Tensor4D::printSize(){
	size.print();
}

double Tensor4D::Max(){
    double res = - 999999999.0;
    for(int i = 0; i < size.len1; i++)
        for(int j = 0; j < size.len2; j++)
            for(int k = 0; k < size.len3; k++)
                for(int l = 0; l < size.len4; ++l)
                    res = max(res, data[i][j][k][l]);
    return res;
}

double Tensor4D::Min(){
    double res = 999999999.0;
    for(int i = 0; i < size.len1; i++)
        for(int j = 0; j < size.len2; j++)
            for(int k = 0; k < size.len3; k++)
                for(int l = 0; l < size.len4; ++l)
                    res = min(res, data[i][j][k][l]);
    return res;
}

void Tensor4D::printMaxMin(){
    cout << "max: " << Max() << ", min: " << Min() << endl;
}

void Tensor4D::cutoff(){
    for(int i = 0; i < size.len1; i++)
        for(int j = 0; j < size.len2; j++)
            for(int k = 0; k < size.len3; k++)
                for(int l = 0; l < size.len4; ++l)
                    data[i][j][k][l] = abs(data[i][j][k][l]) > EPSILON ? data[i][j][k][l] : EPSILON;
}

void Tensor4D::saveData(string filename){
	std::ofstream fp(filename, std::ios::out | std::ios::trunc); 
    assert(fp.is_open());
	for(int i = 0; i < size.len1; ++i){
        for(int j = 0; j < size.len2; ++j){
            for(int k = 0; k < size.len3; ++k) {
                for(int l = 0; l < size.len4 - 1; ++l)
                    fp << data[i][j][k][l] << " "; // ", ";
                fp << data[i][j][k][size.len4 - 1] << endl;
            }
		}
	}
	fp.close();
}

void Tensor4D::loadData(string filename){
	std::fstream fp(filename, std::ios::in);
	assert(fp.is_open());
    double tmp;
    for(int i = 0; i < size.len1; ++i){
        for(int j = 0; j < size.len2; ++j){
            for(int k = 0; k < size.len3; ++k) {
                for(int l = 0; l < size.len4; ++l){
                    fp >> tmp;
                    data[i][j][k][l] = tmp;
                }
            }
		}
	}
	fp.close();
}

void Tensor4D::NewATensor(TensorShape size){
    this->size = size;
	data = new double***[size.len1];
    for(int i = 0; i < size.len1; i++){
        data[i] = new double**[size.len2];
        for(int j = 0; j < size.len2; j++){
            data[i][j] = new double*[size.len3];
            for(int k = 0; k < size.len3; ++k)
                data[i][j][k] = new double[size.len4];
        } 
    }
}

Tensor4D& Tensor4D::operator=(const Tensor4D& Other){
    TensorOpTensor_(Other, op_assign<double>());
    return *this;
}

template<typename T, typename Op> 
Tensor4D& Tensor4D::scalarOpTensor_(T &a, Op op){
    for(int i = 0; i < size.len1; i++)
        for(int j = 0; j < size.len2; j++)
            for(int k = 0; k < size.len3; k++)
                for(int l = 0; l < size.len4; ++l)
                    data[i][j][k][l] = op(a, data[i][j][k][l]);
    return *this;
}

template<typename Op>
Tensor4D& Tensor4D::TensorOpTensor_(const Tensor4D &Other, Op op){
    assert(size == Other.size);
    for(int i = 0; i < size.len1; i++)
        for(int j = 0; j < size.len2; j++)
            for(int k = 0; k < size.len3; k++)
                for(int l = 0; l < size.len4; ++l)
                    data[i][j][k][l] = op(data[i][j][k][l], Other.data[i][j][k][l]);
    return *this;
}

template<typename T, typename Op> 
void Tensor4D::scalarOpTensor(T &a, Tensor4D &output, Op op){
    for(int i = 0; i < size.len1; i++)
        for(int j = 0; j < size.len2; j++)
            for(int k = 0; k < size.len3; k++)
                for(int l = 0; l < size.len4; ++l)
                    output.data[i][j][k][l] = op(a, data[i][j][k][l]);
}

template<typename Op>
void Tensor4D::TensorOpTensor(const Tensor4D &Other, Tensor4D &output, Op op){
    assert(size == Other.size);
    for(int i = 0; i < size.len1; i++)
        for(int j = 0; j < size.len2; j++)
            for(int k = 0; k < size.len3; k++)
                for(int l = 0; l < size.len4; ++l)
                    output.data[i][j][k][l] = op(data[i][j][k][l], Other.data[i][j][k][l]);
}

template<typename Op> 
Tensor4D& Tensor4D::pointwiseFun_(Op op){
    for(int i = 0; i < size.len1; i++)
        for(int j = 0; j < size.len2; j++)
            for(int k = 0; k < size.len3; k++)
                for(int l = 0; l < size.len4; ++l)
                    data[i][j][k][l] = op(data[i][j][k][l]);
    return *this;
}

template<typename Op> 
void Tensor4D::pointwiseFun(const Tensor4D &output, Op op){
    for(int i = 0; i < size.len1; i++)
        for(int j = 0; j < size.len2; j++)
            for(int k = 0; k < size.len3; k++)
                for(int l = 0; l < size.len4; ++l)
                    output.data[i][j][k][l] = op(data[i][j][k][l]);
}

template<typename T, typename Op> 
Tensor4D scalarOpTensor(T &a, const Tensor4D &B, Op op){
    Tensor4D C(B.size);
    for(int i = 0; i < B.size.len1; i++)
        for(int j = 0; j < B.size.len2; j++)
            for(int k = 0; k < B.size.len3; k++)
                for(int l = 0; l < B.size.len4; ++l)
                    C.data[i][j][k][l] = op(B.data[i][j][k][l], a);
    return C;
}

template<typename Op>
Tensor4D TensorOpTensor(const Tensor4D &A,const Tensor4D &B, Op op){
    assert(A.size == B.size);
    Tensor4D C(B.size);
    for(int i = 0; i < B.size.len1; i++)
        for(int j = 0; j < B.size.len2; j++)
            for(int k = 0; k < B.size.len3; k++)
                for(int l = 0; l < B.size.len4; ++l)
                    C.data[i][j][k][l] = op(B.data[i][j][k][l], A.data[i][j][k][l]);
    return C;
}



#endif