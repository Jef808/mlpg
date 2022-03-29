#ifndef TEMPLATEANN_H_
#define TEMPLATEANN_H_

#include <vector>
#include "Activations.h"
#include "Config.h"

#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "Eigen/Core"


namespace simple {


class ANN {
public:
    ANN() = default;

    /**
     * Constructs the data-structure of the model before
     * it can be used for training.
     */
    void setup(Config);

    /**
     *
     */
    void Train(const Eigen::Index batch_size, const double* train_x, const Eigen::Index Ntrain_x,
               const double* train_y, const Eigen::Index Ntrain_y);

    /**
     * Compute the errors with the current model with respect to the given test
     * data @test_x and @test_y.
     */
    template<typename Derived, typename OtherDerived>
    void Predict(const Eigen::MatrixBase<Derived>& inputs, Eigen::MatrixBase<OtherDerived> const& outputs);

    /**
     *
     */
    template <typename Derived, typename OtherDerived>
    void Forward(const Eigen::MatrixBase<Derived>& inputs, Eigen::MatrixBase<OtherDerived> const& outputs);

    /**
     * Runs a backpropagation pass, upgrading the weights
     * from knowledge of the last forward pass.
     *
     * @target is the expected output coming from the data,
     * @gradients output the gradient nabla_W C of the cost function
     * viewed as having fixed input and output, with variable weights
     */
    void Backward();

    /**
     * Calculates the Output layer's error functions in terms of Expected/Obtained outputs
     *
     * @targets The true outputs coming from the dataset
     * @predictions The predictions our current model gives
     * @errors The value of the loss function in terms of the above
     */
    template<typename Derived, typename OtherDerived>
    void Error(const Eigen::MatrixBase<Derived>& targets, const Eigen::MatrixBase<Derived>& predictions,
               Eigen::MatrixBase<OtherDerived> const& errors);

    /**
     * Propagate the error backwards and compute the gradients of the Error.
     *
     * @errors The errors computed from observation at the output layer
     * @gradients The gradients of the Cost function with respect to the current weights
     *
     * @Note For each layer, compute the deltas, i.e. the <partial C(a^L) / partial z^l>
     * when C is seen as having fixed weights and output, with variable input
     */
    template<typename Derived>
    void BackpropagateError(const Eigen::MatrixBase<Derived>& errors);

    void CalculateGradients();

    /**
     * For each layer, update the weights in the direction of @gradients with step size
     * given by @learning_rate.
     *
     * @gradients the gradients of the functional we are trying to minimize
     * @weights The updated weights
     */
    void UpdateWeights();



    [[nodiscard]] const Config& get_config() const;
    /**
     *
     */
    void print(std::ostream& /* output stream */) const;

    /**
     *
     */
    void print_config(std::ostream& /* output stream */) const;

private:
    // Local record of the number of layers in the network
    Eigen::Index n_layers;

    // The number of actual inputs per input nodes
    Eigen::Index n_batches;

    // The current shape of the state, as set by the configuration
    std::vector<Eigen::Index> layout;

    // Controls the step size when updating the network's state.
    double learning_rate;

    // The current state of the network
    std::vector<Eigen::MatrixXd> weights;

    // The current value of the state's neurons' values
    std::vector<Eigen::MatrixXd> layers;

    // The previous value of the state's neurons' values
    std::vector<Eigen::MatrixXd> cache;

    // The current estimate for each neuron's responsibility in the error
    std::vector<Eigen::MatrixXd> deltas;

    // The gradient of the cost function with respec to the weights.
    std::vector<Eigen::MatrixXd> gradients;

    Config m_config;

};  // ANN

/**
 * The layers are still rows, but now these rows are
 * stacked into a number of columns
 */
void ANN::setup(Config config)
{
    m_config = std::move(config);

    // Clear the state
    layout.clear();
    layers.clear();
    cache.clear();
    deltas.clear();
    weights.clear();
    gradients.clear();

    Eigen::Index BS = m_config.batch_size;

    n_layers = static_cast<Eigen::Index>(m_config.HiddenLayers.size()) + 2;
    learning_rate = m_config.LearningRate;

    auto input_size = layout.emplace_back(m_config.InputSize);
    layers.emplace_back(input_size + 1, BS) <<
        Eigen::MatrixXd::Zero(input_size, BS), Eigen::MatrixXd::Constant(1, BS, 1.0);

    for (Eigen::Index i = 0; i < n_layers - 2; ++i) {
        auto output_size = m_config.HiddenLayers[i];

        layers.emplace_back(output_size + 1, BS) <<
            Eigen::MatrixXd::Zero(output_size, BS), Eigen::MatrixXd::Constant(1, BS, 1.0);
        cache.emplace_back(output_size, BS) << Eigen::MatrixXd::Zero(output_size, BS);
        deltas.emplace_back(output_size, BS) << Eigen::MatrixXd::Zero(output_size, BS);
        weights.emplace_back(output_size, input_size + 1)
            << Eigen::MatrixXd::Random(output_size, input_size + 1);  // Still one set of weights
        gradients.emplace_back(output_size, input_size + 1)
            << Eigen::MatrixXd::Zero(output_size, input_size + 1);
        input_size = layout.emplace_back(output_size);
    }

    input_size = m_config.HiddenLayers[n_layers - 3];
    auto output_size = m_config.OutputSize;
    layers.emplace_back(output_size, BS) << Eigen::MatrixXd::Zero(output_size, BS);
    deltas.emplace_back(output_size, BS) << Eigen::MatrixXd::Zero(output_size, BS);
    cache.emplace_back(output_size, BS) << Eigen::MatrixXd::Zero(output_size, BS);
    weights.emplace_back(output_size, input_size + 1) << Eigen::MatrixXd::Random(output_size, input_size + 1);
    gradients.emplace_back(output_size, input_size + 1) << Eigen::MatrixXd::Zero(output_size, input_size + 1);
}


template <typename Derived, typename OtherDerived>
void ANN::Forward(const Eigen::MatrixBase<Derived>& inputs,
                  Eigen::MatrixBase<OtherDerived> const& outputs_)
{
    //SPDLOG_DEBUG("Starting Forward pass");

    // assign inputs to the first layer
    layers[0].topRows(inputs.rows()) = inputs;
    const Eigen::Index L = n_layers - 1;
    //SPDLOG_DEBUG("Inserted `input` into layer 0");

    for (Eigen::Index i = 0; i < n_layers - 1; ++i) {
        // Cache the weighted input values
        cache[i] = weights[i] * layers[i];
        //SPDLOG_DEBUG("Computed weighted inputs for layer {}", i);
        // Compute the activated neuron values
        if (i < n_layers - 2)
            ActivationFunction(cache[i], layers[i + 1].topRows(cache[i].rows()));
        else
            layers[i+1].topRows(cache[i].rows()) = cache[i];
        //SPDLOG_DEBUG("Computed activations for layer {}", i);
    }

    const_cast<Eigen::MatrixBase<OtherDerived>&> (outputs_) = layers[L];
}


template<typename Derived, typename OtherDerived>
void ANN::Error(const Eigen::MatrixBase<Derived>& targets,
                const Eigen::MatrixBase<Derived>& predictions,
                Eigen::MatrixBase<OtherDerived> const& errors_)
{
    //DerActivationFunction(cache.back(), cache.back());
    const_cast<Eigen::MatrixBase<OtherDerived>&> (errors_) = predictions - targets;
}

template<typename Derived>
void ANN::BackpropagateError(const Eigen::MatrixBase<Derived>& errors) {
    deltas.back() = errors;

    // for (Eigen::Index i = n_layers - 2; i > 0; --i) {
    //     DerActivationFunction(cache[i-1], cache[i-1]);
    //     deltas[i-1] = (weights[i].transpose() * deltas[i]).topRows(deltas[i-1].rows()).cwiseProduct(cache[i-1]);
    // }
    for (Eigen::Index i = n_layers - 2; i > 0; --i) {
        DerActivationFunction(cache[i-1], cache[i-1]);
        deltas[i-1] = (weights[i].transpose() * deltas[i]).topRows(deltas[i-1].rows()).cwiseProduct(cache[i-1]);
    }
}

void ANN::CalculateGradients()
{
    for (Eigen::Index i = 0; i < n_layers - 1; ++i)
        gradients[i] = (deltas[i] * layers[i].transpose()) / static_cast<double>(m_config.batch_size);
}


void ANN::UpdateWeights()
{
    // Scale the step size by learning_rate, and gradient really is
    // average over all samples in the batch (the second number)
    for (Eigen::Index i = n_layers - 2; i >= 0; --i) {
        weights[i] -= (learning_rate * gradients[i]);
    }
}

template<typename Derived, typename OtherDerived>
void ANN::Predict(const Eigen::MatrixBase<Derived>& inputs, Eigen::MatrixBase<OtherDerived> const& outputs_)
{
    Forward(inputs, outputs_);
}

void ANN::print(std::ostream& out) const
{
    for (int i=0; i < n_layers - 1; ++i) {
        out << "Layer " << i << "\n\n";
        out << "\nweights:\n" << weights[i];
        out << std::endl;
    }
}


}  // namespace simpleT

#endif  // TEMPLATEANN_H_
