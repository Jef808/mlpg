#ifndef LINEARANN_H_
#define LINEARANN_H_

#include <Eigen/Core>

namespace simple {



struct Config {
    std::size_t InputSize{ 4 };
    std::size_t OutputSize{ 1 };
    std::size_t NLayers{ };
};



class myANN {
public:
    using Vec = Eigen::VectorXf;
    using Mat = Eigen::MatrixXf;

    myANN() = default;

    void setup(Config&& config);

    void Forward(const Vec& input);

    void SampleCost(const Vec& target, float& Cost);

    void Backward(const Vec& target);

    void update();

    template<typename Derived>
    void train(Eigen::DenseBase<Derived> data);

private:

    size_t layout;
    Eigen::MatrixXf weights;
    int InputSize = 4;
    int OutputSize = 1;
    int n_layers = 4;
    float learning_rate;
    Config config;
};




}

#endif // LINEARANN_H_
