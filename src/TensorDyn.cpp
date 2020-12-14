//
// Created by virginie on 26/07/2020.
//

#include "../include/TensorDyn.h"

TensorDyn::TensorOptions TensorDyn::makeTensorWithDims(const std::size_t& i,
                                                       const std::initializer_list<int>& dimensions) const
{
    int * Arr= new int[i];
    std::copy(std::begin(dimensions), std::end(dimensions), Arr);
    switch (i) {

       case 2: {
            Eigen::Tensor<double, 2> T2;
            T2.resize(Arr);
            return T2;
        }
        case 3: {
            Eigen::Tensor<double, 3> T3;
            T3.resize(Arr);
            return T3;
        }
        case 4: {
            Eigen::Tensor<double, 4> T4;
            T4.resize(Arr);
            return T4;
        }
        case 5: {
            Eigen::Tensor<double, 5> T5;
            T5.resize(Arr);
            return T5;
        }
    }
    delete [] Arr;
}

// Getter
TensorDyn::TensorOptions TensorDyn::get() const {
    return TnsDyn;
}
std::initializer_list<int> TensorDyn::getDims() const {
    return dims;
}

std::size_t TensorDyn::getRank() const {
    return rnk;
}

std::ostream &operator<<(std::ostream &os, const TensorDyn &dyn) {
    os << "rnk: " << dyn.rnk <<  " TnsDyn: " << dyn.TnsDyn;
    return os;
}

// Setter
/*
void TensorDyn::readCSV(const std::string &path) {
    Eigen::MatrixXd MatToRead = load_csv<Eigen::MatrixXd>(path);
    auto MappedTens = Matrix_to_Tensor(MatToRead,dims);
    TnsDyn=TensorLayoutSwap(MappedTens);
} */



