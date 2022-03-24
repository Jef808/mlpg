#ifndef LINEARANN_H_
#define LINEARANN_H_


#include <iosfwd>
#include <vector>

#include "Eigen/Dense"

namespace simple {


  struct Config { using Index = Eigen::Index;
     Index InputSize; Index OutputSize; std::vector<Index> HiddenLayers; double LearningRate; double MomentumGain;
  };


class myANN {

public: using Index = Config::Index; myANN() = default;

    /**
     * Constructs the data-structure of the model before
     * it can be used for training.
     */
    void setup(Config);

    /**
     *
     */
    void Train(const Eigen::MatrixXd& train_x, const Eigen::MatrixXd& train_y);

    /**
     *
     */
    void Predict(const Eigen::MatrixXd& test_x, Eigen::MatrixXd& test_y);

    /**
     * Runs a forward pass, where each node's
     * value is computed according to the
     * given `input` and the current `policy`
     * of the network.
     */
    void Forward(const Eigen::VectorXd& input);

    /**
     * For each layer, compute the deltas, i.e. the <partial C(a^L) / partial z^l>
     * when C is seen as having fixed weights and output, with variable input
     */
    void BackpropagateError(const Eigen::VectorXd& target);

    /**
     * Runs a backpropagation pass, upgrading the weights
     * from knowledge of the last forward pass.
     *
     * @target is the expected output coming from the data,
     * @gradients contains the gradient nabla_W C of the cost function
     * viewed as having fixed input and output, with variable weights
     */
    void Backward(const Eigen::VectorXd& target, std::vector<Eigen::MatrixXd>& gradient);

    /**
     * For each layer, update the weights in the direction of @gradients with step size
     * given by @learning_rate.
     */
    void UpdateWeights(Index minibatch_size = 1);

    double AverageLoss(const Eigen::MatrixXd& predictions, const Eigen::MatrixXd& targets);

    /**
     *
     */
    std::vector<Index> get_layout();

    /**
     *
     */
    void print(std::ostream& /* output stream */);

    /**
     *
     */
    void print_config(std::ostream& /* output stream */);

private:
    // Local record of the number of layers in the network
    Index n_layers;

    // The current shape of the state, as set by the configuration
    std::vector<Index> layout;

    // Controls the step size when updating the network's state.
    double learning_rate;

    // Controls the effective (adaptative) learning rate
    double momentum_gain;

    // The current state of the network
    std::vector<Eigen::MatrixXd> weights;

    // The current value of the state's neurons' values
    std::vector<Eigen::VectorXd> layers;

    // The previous value of the state's neurons' values
    std::vector<Eigen::VectorXd> cache;

    // The current estimate for each neuron's responsibility in the error
    std::vector<Eigen::VectorXd> deltas;

    // The gradient of the cost function with respec to the weights.
    std::vector<Eigen::MatrixXd> gradients;

    Config m_config;

  };  // myANN



}  // namespace simple

#endif  // LINEARANN_H_
