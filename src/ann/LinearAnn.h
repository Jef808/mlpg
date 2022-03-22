#ifndef LINEARANN_H_
#define LINEARANN_H_


#include <iosfwd>
#include <vector>

#include "Eigen/Dense"

namespace simple {


  struct Config { using Index = Eigen::Index;
     Index InputSize; Index OutputSize; std::vector<Index> HiddenLayers;
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
    void Train();

    /**
     * Runs a forward pass, where each node's
     * value is computed according to the
     * given `input` and the current `policy`
     * of the network.
     */
    void Forward(const Eigen::VectorXd& input);

    /**
     * Runs a backpropagation pass, upgrading the weights
     * from knowledge of the last forward pass.
     */
    void Backward(const Eigen::VectorXd& target);

    /**
     *
     */
    void CalculateErrors(const Eigen::VectorXd& target);

    /**
     *
     */
    void update();

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

    // The current state of the network
    std::vector<Eigen::MatrixXd> weights;

    // The current value of the state's neurons' values
    std::vector<Eigen::VectorXd> layers;

    // The previous value of the state's neurons' values
    std::vector<Eigen::VectorXd> cache;

    // The current estimate for each neuron's responsibility in the error
    std::vector<Eigen::VectorXd> deltas;

    Config m_config;

  };  // myANN



}  // namespace simple

#endif  // LINEARANN_H_
