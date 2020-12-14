//
// Created by virginie on 9/08/2020.
//

#include "Eigen/CXX11/Tensor"
#include "include/TAJIVE.h"
#include <type_traits>
#include <ctime>

template<typename M>
M load_csv (const std::string & path) {
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

template < typename T >
decltype(auto) TensorLayoutSwap(T&& t)
{
    return Eigen::TensorLayoutSwapOp<typename std::remove_reference<T>::type>(t);
}


template<class... Ts> struct overload : Ts... { using Ts::operator()...; };

template<class... Ts> overload(Ts...) -> overload<Ts...>;



int main() {
    Eigen::MatrixXd A = load_csv<Eigen::MatrixXd>("/home/virginie/Desktop/VITALE/Tesi/Codice/TAJIVEc++/data/Xtoy.csv");
    Eigen::MatrixXd B = load_csv<Eigen::MatrixXd>("/home/virginie/Desktop/VITALE/Tesi/Codice/TAJIVEc++/data/Ytoy.csv");

    /*   // Approach to convert the matrices A and B into a 2D-tensor ---> failed
       Eigen::Tensor<double,2> TensA =  Matrix_to_Tensor(A, 100,1000);
       Eigen::Tensor<double,2> TensB =  Matrix_to_Tensor(B, 1000,1000);

      auto TensLayA=TensorLayoutSwap(TensA);
      auto  TensLayB=TensorLayoutSwap(TensB);

       TensorDyn TendDynA= static_cast<const TensorDyn::TensorOptions &>(TensLayA);
       TensorDyn TendDynB=TensLayB;

       std::vector<TensorDyn> ToyDataset={TensA,TensB};
      TAJIVE ToyTest(ToyDataset,1); */

    // Common dimension is the first one: mode-0
    std::vector<Eigen::MatrixXd> data = {A.block(0, 0, 40, 20), B.block(0, 0, 40, 40)};

    // Estimated rank with a scree plot
    const std::vector<std::vector<std::size_t>> ranks = {{2},
                                                         {3}};

    TAJIVE ToyTest(data, ranks);

    // method intentionally made public in order to be able to be called here

    time_t tstart, tend;
    tstart = time(0);
    ToyTest.SignalSpaceExtraction(0);

    ToyTest.JointSpaceEstimation(0);

    ToyTest.FinalExtraction(0);
    tend = time(0);
    long double elapsed=difftime(tend, tstart);
    std::cout << "It took "<< elapsed <<" second(s)."<< std::endl;


    return 0;
}


