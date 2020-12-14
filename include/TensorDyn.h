//
// Created by virginie on 26/07/2020.
//

#ifndef TAJIVEDECOMP_TENSORDYN_H
#define TAJIVEDECOMP_TENSORDYN_H

#include <iostream>
#include "Eigen/CXX11/Tensor"
#include "../dependecies/Eigen/Dense"

#include <utility>
#include <memory>
#include <variant>
#include <iterator>     // std::advance
#include <vector>
#include <list>
#include <fstream>
#include <random>


#define FUNC_SINGLE_RETURN(Method, Tensor, ReturnVar) do {\
            ReturnVar = std::visit([](const auto &tensor) { \
            return tensor.Method;}, Tensor.get()); } while (0)

class TensorDyn {
public:
    typedef std::variant< Eigen::Tensor<double, 2> , Eigen::Tensor<double, 3>,
            Eigen::Tensor<double, 4>, Eigen::Tensor<double, 5>> TensorOptions;
    template<typename T>
    using  MatrixType = Eigen::Matrix<T,Eigen::Dynamic, Eigen::Dynamic>;

private:
    std::size_t rnk=0;
    const std::initializer_list<int> dims;

public:
    TensorOptions TnsDyn;
    // Constructor where the sizes for the constructor are specified as an array of values
    TensorDyn(const std::size_t& rnk, const std::initializer_list<int>& dims):rnk(rnk),dims(dims){
        TnsDyn=makeTensorWithDims(rnk,dims);
    }

    // Constructor for readCSV
    TensorDyn(const TensorOptions& tns){TnsDyn=tns;};

    // Method to call in the constructor
    TensorOptions makeTensorWithDims(const std::size_t& i,const std::initializer_list<int>& dimensions) const;

    //Getter
    TensorOptions get() const;
    std::initializer_list<int> getDims() const;
    std::size_t getRank() const;

    friend std::ostream &operator<<(std::ostream &os, const TensorDyn &dyn);

    // Setter
    /*    void readCSV(const std::string & path);
       template<typename M>
       M load_csv (const std::string & path);
       template<typename Scalar, typename... Dims>
       auto MatrixToTensor(const MatrixType<Scalar> &matrix, std::initializer_list<Dims>... dims);
      template < typename T >
       TensorOptions TensorLayoutSwap(T&& t); */

};
/*
template<typename M>
M TensorDyn::load_csv (const std::string & path) {
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<double> values;
    uint rows = 0;
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    return Eigen::Map<const Eigen::Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, Eigen::RowMajor>>(values.data(), rows, values.size()/rows);
}

template<typename Scalar, typename... Dims>

auto TensorDyn::MatrixToTensor(const MatrixType<Scalar> &matrix, std::initializer_list<Dims>... dims)
{
    constexpr int rank = sizeof... (Dims);
    return Eigen::TensorMap<Eigen::Tensor<const Scalar, rank>>(matrix.data(), dims...);
}

template < typename T >
decltype(auto) TensorDyn::TensorLayoutSwap(T&& t)
{
    return Eigen::TensorLayoutSwapOp<typename std::remove_reference<T>::type>(t);
} */

#endif //TAJIVEDECOMP_TENSORDYN_H
