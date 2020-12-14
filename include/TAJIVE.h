//
// Created by virginie on 07/08/2020.
//

#ifndef TAJIVEC_TAJIVEDECOMP_H
#define TAJIVEC_TAJIVEDECOMP_H

#include "TensorDyn.h"
#include "../dependecies/Eigen/Dense"
#include "../dependecies/Eigen/SVD"

#include <armadillo>

#include <omp.h>
#include <algorithm>
#include <limits>
#include <chrono>
#include <random>
#include <utility>
#include <vector>
#include <set>
#include <memory>
#include <math.h>

#define nresample 100   // number of simulation
#define percentile 5    // Percentile of perturbation bounds used for deciding the joint rank
#define err pow(10,-10) // To add to singular value to avoid floating error

class TAJIVE {
public:
    struct TAJIVEsegm{
        std::vector<std::vector<TensorDyn> > JointSignals;
        std::vector<std::vector<TensorDyn> > IndSignals;
        std::vector<std::vector<TensorDyn> > Noise;
        std::vector<std::vector<std::size_t> > jointRank;
        std::vector<std::vector<std::size_t> > indRank;
    };

private:
    // ####  Members to define with constuctor  ###
    std::vector<TensorDyn> data_blocks;           // Data-sets
    std::vector<Eigen::MatrixXd> data_blocks_mat; // Data-sets when only matrices

    std::size_t K;                                // number of data-sets
    std::vector<std::vector<std::size_t>> rank_estimates; // estimated rank for each dataset [dataset<dims>]
    std::size_t N;                               // number common modes
    bool areAllMats;                             // to use data_blocks_mat instead of data_blocks


    // ####  Members for the analysis  ###
    Eigen::MatrixXd StackMat;                   // Matrix with the stacked right singular matrix of each data block.
    Eigen::MatrixXd AnglesBound;                // In radians
    std::vector<double> Thresholds;             // for singular values
    Eigen::MatrixXd JointMat;                   // Joint column space
    std::size_t jointRank;                      // joint rank
public:
    // ####  Constructors  ###
    // Constructor in case of data-sets with also Tensors

    TAJIVE(const std::vector<TensorDyn> &dataBlocks, const  std::size_t& _N,
           const std::vector<std::vector<std::size_t>>& rnkEstim);


    // Constructor in case of Matrices data-sets of only Matrices
    TAJIVE(const std::vector<Eigen::MatrixXd> &dataBlocks, const std::vector<std::vector<std::size_t>>&  rnkEstim);

    // ####  Main methods for the analysis ###

    // Step 1: Signal Space Initial Extraction
     void SignalSpaceExtraction(const std::size_t nC);
    // Step 2: Joint Score Space Estimation
    std::size_t computeRank(const Eigen::VectorXd& vec, const double bound) ; // da  parallelizz
    void JointSpaceEstimation(const std::size_t nC);
    // Step 3: Final Decomposition
    void FinalExtraction(const std::size_t nC);
    // Method which calls all the Steps and print Outputs
    TAJIVEsegm TAJIVEdec();

private:

    // ####  Methods to support the analysis ###
    friend TensorDyn;

    // Unfolding a tensor wrt specific mode
    friend Eigen::MatrixXd unfolding(TensorDyn& T,const std::size_t idx);

    // Resampled perturbation principal angle bounds of SVD approximation
    friend std::vector<double> L2normNoisyDir(const Eigen::MatrixXd & Data, const Eigen::MatrixXd& ColSpace,const int numDir);

    // Template functions to convert a Eigen::Matrix to an Eigen::Tensor and viceversa
    template<typename T>
    using  MatrixType = Eigen::Matrix<T,Eigen::Dynamic, Eigen::Dynamic>;

    template<typename Scalar, typename... Dims>
    auto Matrix_to_Tensor(const MatrixType<Scalar> &matrix, Dims... dims);

    template<typename Scalar,int rank, typename sizeType>
    auto Tensor_to_Matrix(const Eigen::Tensor<Scalar,rank> &tensor,const sizeType rows,const sizeType cols);

    // ### Getter ###
public:
    const Eigen::MatrixXd &getStackMat() const;

    const Eigen::MatrixXd &getAnglesBound() const;

    const std::vector<double> &getThresholds() const;

    const Eigen::MatrixXd &getJointMat() const;

};

// ####  Other methods to support the analysis ###

//compute percentile
double prctile(const Eigen::VectorXd& vec, const int prc); // WORKS CORRETLY

// Empirical distribution of largest squared singular value
Eigen::VectorXd EmpDistrLargSV(const std::size_t lenghtDim, const std::vector<std::size_t>& dimsSubsp );

// Remove Column from a Eigen::Matrix
void removeColumn(Eigen::MatrixXd& matrix, const std::set<int>& colToRemove); // WORKS CORRETLY

// Random matrices
Eigen::MatrixX<double> random_matrix(int rows, int cols, double min, double max, unsigned seed);


// ####  Friend methods ###

Eigen::MatrixXd unfolding(const TensorDyn& T,const std::size_t idx); // WORKS CORRETLY

std::vector<double> L2normNoisyDir(const Eigen::MatrixXd & Data, const Eigen::MatrixXd& ColSpace,const int numDir); // PROBLEM WITH generating random matrices

template<typename T>
using  MatrixType = Eigen::Matrix<T,Eigen::Dynamic, Eigen::Dynamic>;

template<typename Scalar, typename... Dims>
auto Matrix_to_Tensor(const MatrixType<Scalar> &matrix, Dims... dims)
{
    constexpr int rank = sizeof... (Dims);
    return Eigen::TensorMap<Eigen::Tensor<const Scalar, rank>>(matrix.data(), {dims...});
}

template<typename Scalar,int rank, typename sizeType>
auto Tensor_to_Matrix(const Eigen::Tensor<Scalar,rank> &tensor,const sizeType rows,const sizeType cols)
{
    return Eigen::Map<const MatrixType<Scalar>> (tensor.data(), rows,cols);
}

#endif //TAJIVEC_TAJIVEDECOMP_H
