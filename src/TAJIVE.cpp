//
// Created by virginie on 07/08/2020.
//

#include "../include/TAJIVE.h"

// ####  Constructors  ###

TAJIVE::TAJIVE(const std::vector<TensorDyn> &dataBlocks, const  std::size_t& _N,
               const std::vector<std::vector<std::size_t>>& rnkEstim) :
        data_blocks(dataBlocks),N(_N),K(dataBlocks.size()),rank_estimates(rnkEstim) {
    areAllMats= false;
    AnglesBound.resize(K,nresample);
    Thresholds.reserve(K);

}

TAJIVE::TAJIVE(const std::vector<Eigen::MatrixXd> &dataBlocks,const  std::vector<std::vector<std::size_t>> & rnkEstim) :
data_blocks_mat(dataBlocks),K(dataBlocks.size()),rank_estimates(std::move(rnkEstim)) {
    std::cout<<"TAJIVE decomposition with "<<K<<" datasets"<<std::endl;
    std::cout<<"The initial estimated ranks are respectively"<<std::endl;
    for(const auto& data:rank_estimates){
        for (const auto&x:data)
            std::cout<<x<<" ";
        std::cout<<"\n";
    }
    areAllMats=true;
    N=1;
    AnglesBound.resize(K,nresample);
    Thresholds.reserve(K);
 }

// ####  Main methods for the analysis ###
/*
TAJIVE::TAJIVEsegm TAJIVE::TAJIVEdec() {

     std::cout <<"RECONSTRUCTION OF THE TAJIVE DECOMPOSITION"<<std::endl;

    // Reconstruction the TAJIVE decomposition in each datablock

    std::vector<std::vector<TensorDyn> > JointSignals(K,std::vector<TensorDyn> (N));
    std::vector<std::vector<TensorDyn> > IndSignals(K,std::vector<TensorDyn> (N));
    std::vector<std::vector<TensorDyn> > Noise(K,std::vector<TensorDyn> (N));
    std::vector<std::size_t>  FinalJointRank(N);
    std::vector<std::vector<std::size_t> > FinalIndRank(K,std::vector<std::size_t> (N));

    for (int iN = 0; iN < N; ++iN) {
    SignalSpaceExtraction(iN);
    JointSpaceEstimation(iN);
    FinalExtraction(iN);
    FinalJointRank[iN]=jointRank;
        for (int iK = 0; iK < K; ++iK) {
            Eigen::MatrixXd DataUnfold=areAllMats==0?unfolding(data_blocks[iK], iN):data_blocks_mat[iK];
            Eigen::MatrixXd JointSignalUnfolded=DataUnfold*JointMat.transpose()*JointMat;

            Eigen::MatrixXd IndSignalUnfoldedNotReduced=DataUnfold-JointSignalUnfolded;
            Eigen::BDCSVD<Eigen::MatrixXd> svd(IndSignalUnfoldedNotReduced, Eigen::ComputeFullU | Eigen::ComputeFullV );
            Eigen::MatrixXd Smat=svd.singularValues();
            Eigen::VectorXd Sfull(Eigen::Map<Eigen::VectorXd>(Smat.data(),Smat.rows()));

            FinalIndRank[iK][iN]= computeRank(Sfull.array().square() , Thresholds[iK]);
            Eigen::MatrixXd Uind=svd.matrixU().block(0,0,svd.matrixU().rows(),FinalIndRank[iK][iN]);
            Eigen::MatrixXd Dind=svd.singularValues().block(0,0,FinalIndRank[iK][iN],FinalIndRank[iK][iN]);
            Eigen::MatrixXd Vind=svd.matrixV().block(0,0,svd.matrixV().rows(),FinalIndRank[iK][iN]);

            Eigen::MatrixXd IndSignalUnfolded=Uind*Dind*Vind.transpose();

            Eigen::MatrixXd NoiseUnfolded=JointSignalUnfolded-IndSignalUnfolded;

            if(areAllMats==0){
                std::list<std::size_t> newDim;
                std::copy(data_blocks[iK].getDims().begin()+1, data_blocks[iK].getDims().end(), newDim);
                newDim.push_front(FinalJointRank[iN]);
                JointSignals[iK][iN]=Matrix_to_Tensor(JointSignalUnfolded, newDim);

                newDim.pop_front();
                newDim.push_front(FinalIndRank[iK][iN]);
                IndSignals[iK][iN]=Matrix_to_Tensor(IndSignalUnfolded,newDim);
                Noise[iK][iN]=Matrix_to_Tensor(NoiseUnfolded,data_blocks[iK].getDims());
                data_blocks[iK]=-JointSignals[iK][iN]; // usare std::visit


            } else{
                JointSignals[iK][iN]=JointSignalUnfolded;
                IndSignals[iK][iN]=IndSignalUnfolded;
                Noise[iK][iN]=NoiseUnfolded;
                data_blocks_mat[iK]=-JointSignals[iK][iN];
            }
        }

    }
    struct TAJIVEsegm output = {JointSignals,IndSignals,Noise,FinalJointRank,FinalIndRank};
    return output;
}
*/
void TAJIVE::SignalSpaceExtraction(const std::size_t nC) {
    std::cout <<"SIGNAL SPACE EXTRACTION for mode "<<nC<<std::endl;
    // Unfolding all dataset wrt specific mode (nC)
    std::vector<Eigen::MatrixXd> data_block_unfolded;
    data_block_unfolded.reserve(areAllMats==0?data_blocks.size():0);

    int numCol=0;
    for (auto& rnks:rank_estimates)
        numCol+=rnks[nC];

    if(areAllMats==0) {
        for (auto &data:data_blocks)
            data_block_unfolded.push_back(unfolding(data, nC));
    }
    int lengthnC=areAllMats==0?data_block_unfolded[0].rows():data_blocks_mat[0].rows();

    StackMat.resize(lengthnC,numCol);

    // Low rank svd of each data block
    std::size_t i=0;
    for (int iK = 0; iK < K; ++iK) {
        Eigen::BDCSVD<Eigen::MatrixXd> svd(areAllMats==0?data_block_unfolded[iK]:data_blocks_mat[iK]
                , Eigen::ComputeFullU | Eigen::ComputeFullV);
        const auto & rnk = rank_estimates[iK][nC];
        const Eigen::MatrixXd & U = svd.matrixU().block(0,0,svd.matrixU().rows(),rnk);
        const Eigen::VectorXd & S = svd.singularValues();
        const Eigen::MatrixXd & V = svd.matrixV().block(0,0,svd.matrixV().rows(),rnk);

        // Threshold of singular values
        if (rnk+1 <= svd.rank())
            Thresholds.push_back((S(rnk)+S(rnk+1)/2));
        else
            Thresholds.push_back(S(svd.rank())/2);

        //  Filling stack matrix

        Eigen::MatrixXd toStack(U.leftCols(rnk));
        StackMat.block(0,i,lengthnC,rnk)=toStack;
        i+=rnk;

        //  Estimating approximation accuracy

        std::vector<double> _Rcreatenorm{L2normNoisyDir(areAllMats==0?data_block_unfolded[iK]:data_blocks_mat[iK], V,rnk)};
        Eigen::VectorXd Rcreatenorm = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(_Rcreatenorm.data(), _Rcreatenorm.size());
        std::vector<double> _Screatenorm{L2normNoisyDir(areAllMats==0?data_block_unfolded[iK].transpose():data_blocks_mat[iK].transpose(), U,rnk)};
        Eigen::VectorXd Screatenorm = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(_Screatenorm.data(), _Screatenorm.size());

        double delta{S(rnk)};

        Eigen::VectorXd lambda=(Rcreatenorm.cwiseMax(Screatenorm)/delta).cwiseMin(1);
        Eigen::VectorXd lambdaSin = lambda.unaryExpr(
                [](const double& x) {
                    return asin(x);
                });
        AnglesBound.block(iK,0,1,nresample)= lambdaSin.transpose(); //(lambda * 180) / PI;
    }
}

std::size_t TAJIVE::computeRank(const Eigen::VectorXd& vec, const double bound) {
    std::size_t rnk{0};

    for (size_t j = 0; j < vec.size(); ++j) {
        if (pow(vec(j),2)+err>bound)
            rnk++;
    }
    return rnk;
}


void TAJIVE::JointSpaceEstimation(const std::size_t nC) {
    std::cout <<"JOINT SPACE ESTIMATION for mode "<<nC<<std::endl;

    std::vector<std::size_t> ranksC;
    ranksC.reserve(areAllMats==0?data_blocks.size():data_blocks_mat.size());
    for (const auto& data:rank_estimates)
        ranksC.push_back(data[nC]);

    std::size_t minRank{*(std::min_element(ranksC.cbegin(),ranksC.cend()))};

    // Apply SVD on concatenated matrix
    Eigen::BDCSVD<Eigen::MatrixXd> svd(StackMat, Eigen::ComputeThinU );
    Eigen::MatrixXd Smat=svd.singularValues();
    Eigen::VectorXd Sfull(Eigen::Map<Eigen::VectorXd>(Smat.data(),Smat.rows()));
    Eigen::VectorXd S=Sfull.head(minRank);

    // Threshold based on Wedin Bound

    Eigen::MatrixXd CosAngleBound = AnglesBound.unaryExpr(
            [](const double& x) {
                return pow(cos(x),2);
            });
    Eigen::VectorXd WedinSSVbds=CosAngleBound.colwise().sum();
    Eigen::VectorXd vecr1 = WedinSSVbds.unaryExpr(
            [](const double& x) {
                return x<1?1:x;
            });
    double WedinSSVbd=prctile(vecr1,5);

    // Random directon bound

    Eigen::VectorXd randSSVs=EmpDistrLargSV(StackMat.rows(),ranksC);
    double randSSVbd=prctile(randSSVs,95);

    // Use of the selected bound percentile for rank selection

    if (randSSVbd > prctile(WedinSSVbds,5)){
        std::cout<<"The Wedin bound is too loose that it is smaller than the random direction bound."<<std::endl;
        std::cout<<"This suggests reducing the input rank. Will use random direction bound instead."<<std::endl;
        jointRank= computeRank(S, randSSVbd);
    } else
        jointRank= computeRank(S, WedinSSVbd);

    std::cout<<"Proposed Joint rank: "<<jointRank<<std::endl;

    JointMat=svd.matrixU().block(0,0,StackMat.rows(),jointRank);
}

void TAJIVE::FinalExtraction(const std::size_t nC) {
    // Delete the col in JointMat which have low  variance projected on some datablocks
    std::set<int> ToDelete;
    for (int iK = 0; iK < K; ++iK) {
        Eigen::MatrixXd dataUnfolded = areAllMats == 0 ? unfolding(data_blocks[iK], nC) : data_blocks_mat[iK];
        Eigen::MatrixXd JointDirection{ dataUnfolded.transpose() * JointMat};

        if (JointDirection.rows()>0) {
            Eigen::MatrixXd vecr1 = JointDirection.unaryExpr(
                    [](const double &x) {
                        return pow(x, 2);
                    });
            Eigen::VectorXd LowSdJoint = vecr1.colwise().sum();
            std::vector<Eigen::Index> LowSdJointFound;
            for (Eigen::Index i = 0; i < LowSdJoint.size(); ++i) {
                if (sqrt(LowSdJoint(i)) <= Thresholds[iK] + err)
                    LowSdJointFound.push_back(i);
            }


            if (!LowSdJointFound.empty()) {
                std::cout << "The following basis vector have low variance in the " << iK+1 << "th dataset" << std::endl;
                for (const Eigen::Index i:LowSdJointFound)
                    std::cout << i << std::endl;
                std::copy(LowSdJointFound.begin(), LowSdJointFound.end(),
                          std::inserter(ToDelete, ToDelete.end()));
            }

            removeColumn(JointMat, ToDelete);
            jointRank = JointMat.cols();
        }
    }
    std::cout << "Final Joint rank: " << jointRank << std::endl;
}


Eigen::MatrixXd unfolding( const TensorDyn& T,const std::size_t idx)  {
    // Initial operations to put the indexes in the right order
    std::list<size_t> idxShOrdered(T.getRank());
    std::iota(idxShOrdered.begin(), idxShOrdered.end(), 0);
    auto it=idxShOrdered.begin();
    std::advance (it,idx);
    idxShOrdered.erase(it);
    std::size_t product = std::accumulate(idxShOrdered.begin(), idxShOrdered.end(), 1, std::multiplies<std::size_t>());
    idxShOrdered.push_front(idx);
    // Reshape indexes
    Eigen::array<std::size_t ,2>idxResh={idx,product};
    // Shuffle indexes
    int * idxSh= new int[T.getRank()];
    std::copy(std::begin(idxShOrdered), std::end(idxShOrdered), idxSh);

    // Unfolding
    Eigen::Tensor<double,2> unfolded = std::visit([&idxResh, &idxSh]( auto& tensor) {
        // Shuffle
        auto tensSh=tensor.shuffle(idxSh);
        // Reshape
        Eigen::Tensor<double,2> tensResh=tensSh.reshape(idxResh);
        return tensResh;},T.TnsDyn);

    delete [] idxSh;
    // Convert  Eigen::Tensor<double,2> to  Eigen::MatrixXd
    Eigen::MatrixXd unfoldedMat =  Tensor_to_Matrix(unfolded, idx,product);
    return unfoldedMat;
}

std::vector<double> L2normNoisyDir(const Eigen::MatrixXd & Data, const Eigen::MatrixXd& EstimSpace,const int numDir){
    std::vector<double>  Norm;
    Norm.reserve(nresample);
    std::normal_distribution<double> distribution(0,1);

    // #pragma omp parallel for
    for (int i = 0; i < nresample; ++i) {
        Eigen::MatrixXd EstimSpaceSim(EstimSpace.rows(),EstimSpace.cols()+numDir);
        EstimSpaceSim.block(0,0,EstimSpace.rows(),EstimSpace.cols())=EstimSpace;
        Eigen::MatrixXd nullDir(EstimSpace.rows(),numDir);
        for (int j = 0; j < numDir; ++j) {
            std::default_random_engine engine (nresample * (i+1) * (j+1) );
            Eigen::MatrixXd tmp = Eigen::MatrixXd::NullaryExpr(EstimSpace.rows(),1,[&](){return distribution(engine);});
            Eigen::VectorXd Proj(EstimSpaceSim.transpose()*tmp);
            Eigen::VectorXd tmp2(EstimSpaceSim*Proj);
            Eigen::VectorXd tmpMat(tmp-tmp2);
            tmpMat.normalize();

            nullDir.block(0,j,EstimSpace.rows(),1)=tmpMat;
            EstimSpaceSim.block(0,EstimSpace.cols()+j,EstimSpace.rows(),1)=tmpMat;
        }
        Eigen::BDCSVD<Eigen::MatrixXd> svd(nullDir, Eigen::ComputeThinU );
        // matrixU is the orthonormal basis for nullDir
        double normValue=(Data*svd.matrixU()).norm();
        Norm.push_back(normValue);
    }
    return Norm;
}

const Eigen::MatrixXd &TAJIVE::getStackMat() const {
    return StackMat;
}

const Eigen::MatrixXd &TAJIVE::getAnglesBound() const {
    return AnglesBound;
}

const std::vector<double> &TAJIVE::getThresholds() const {
    return Thresholds;
}

const Eigen::MatrixXd &TAJIVE::getJointMat() const {
    return JointMat;
}

double prctile(const Eigen::VectorXd& vec, const int prc){
    std::vector<double> vStd;
    vStd.reserve(vec.size());
    Eigen::VectorXd::Map(&vStd[0], vec.size()) = vec;
    auto nth = vStd.begin() + (prc*vStd.size())/100;
    std::nth_element(vStd.begin(), nth, vStd.end());
    return *nth;
}

Eigen::VectorXd EmpDistrLargSV(const std::size_t lenghtDim, const std::vector<std::size_t>& dimsSubsp ){
    Eigen::VectorXd randSSVs;
    randSSVs.resize(nresample);

    for (int i = 0; i <nresample ; ++i) {
        arma::mat tmp2;
        for (int iK = 0; iK < dimsSubsp.size(); ++iK) {
           arma::mat tmp(nresample,dimsSubsp[iK]);
           tmp.randn();
           arma::mat Orth=arma::orth(tmp).t();
            std::size_t row_start=std::accumulate(dimsSubsp.begin(),dimsSubsp.begin()+iK,0);
            tmp2.insert_rows(row_start,Orth);
        }
        randSSVs(i)=arma::norm(tmp2,2);
    }
    return randSSVs;
}



void removeColumn(Eigen::MatrixXd& matrix, const std::set<int>& colToRemove)
{
    for (auto rit=colToRemove.rbegin(); rit!=colToRemove.rend(); ++rit){
        unsigned int numRows = matrix.rows();
        unsigned int numCols = matrix.cols()-1;
        if( *rit < numCols )
            matrix.block(0,*rit,numRows,numCols-*rit) = matrix.rightCols(numCols-*rit);

        matrix.conservativeResize(numRows,numCols);
    }
}

Eigen::MatrixX<double> random_matrix(int rows, int cols, double mean, double std, unsigned seed)
{
    std::mt19937 generator(seed);
    std::normal_distribution<double> distribution(mean,std);
    Eigen::MatrixX<double> result(rows,cols);
    for(double& val : result.reshaped())
        val = distribution(generator);
    return result;
}



