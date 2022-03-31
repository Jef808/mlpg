#ifndef TEMPLATEANN_H_
#define TEMPLATEANN_H_

#include "Activations.h"
#include "Config.h"

#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "Eigen/Core"

#include <vector>
#include <type_traits>

namespace simple {

template<typename FT>
class ANN {
public:
    using Matrix = typename std::conditional< std::is_same_v<FT, float>,
                                              Eigen::MatrixXf, Eigen::MatrixXd >::type;

    ANN() = default;

    /**
     * Constructs the data-structure of the model before
     * it can be used for training.
     */
    void setup(Config);

    /**
     *
     */
    void Train(const size_t NEpochs, const Eigen::Index batch_size,
               const FT* train_x, const Eigen::Index Ntrain_x,
               const FT* train_y, const Eigen::Index Ntrain_y);

    /**
     * Compute the errors with the current model with respect to the given test
     * data @test_x and @test_y.
     */
    template<typename Derived, typename OtherDerived>
    void Predict(const Eigen::MatrixBase<Derived>& inputs, Eigen::MatrixBase<OtherDerived> const& outputs);

    /**
     * Feed @inputs into the network and collect the @outputs at the last layer.
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
     * Get the sum of the cost function for a set of targets/outputs.
     *
     * @targets The target values from the dataset, one per column.
     * @outputs The outputs predicted by the model, one per column.
     * @Return The sum of the cost function for each pair of columns.
     */
    template<typename Derived, typename OtherDerived>
    void CostSum(const Eigen::MatrixBase<Derived>& targets, const Eigen::MatrixBase<OtherDerived>& outputs,
                 double& cost);

    /**
     * Calculates the Output layer's deltas in terms of Expected/Obtained outputs
     *
     * @targets The true outputs coming from the dataset
     * @predictions The predictions our current model gives
     * @errors The value of the loss function in terms of the above
     */
    template<typename Derived, typename OtherDerived>
    void Error(const Eigen::MatrixBase<Derived>& targets, const Eigen::MatrixBase<OtherDerived>& predictions,
               Eigen::MatrixBase<OtherDerived> const& errors);

    /**
     * Propagate the error backwards and compute the gradients of the cost function.
     *
     * @errors The errors computed from observation at the output layer
     * @gradients The gradients of the Cost function with respect to the current weights
     *
     * @Note For each layer, compute the deltas, i.e. the <partial C(a^L) / partial z^l>
     * when C is seen as having fixed weights and output, with variable input
     */
    template<typename Derived>
    void BackpropagateError(const Eigen::MatrixBase<Derived>& errors);

    /**
     * Compute the gradient of the cost function with respect to the weights,
     * viewing the inputs and outputs as fixed.
     */
    void CalculateGradients();

    /**
     * For each layer, update the weights in the direction of @gradients with step size
     * given by @learning_rate.
     */
    void UpdateWeights();

    /**
     * Print the current weights of the network to the given output stream.
     */
    void print(std::ostream& /* output stream */) const;

private:
    // The number of layers in the network
    Eigen::Index n_layers;

    // The number of inputs per input nodes
    Eigen::Index n_batches;

    // Controls the step size when updating the network's weights
    double learning_rate;

    // The current shape of the network, as set by the configuration
    std::vector<Eigen::Index> layout;

    // The current weights of the network
    std::vector<Matrix> weights;

    // The current value of the network's activations
    std::vector<Matrix> layers;

    // To save the weighted inputs during a forward pass
    std::vector<Matrix> cache;

    // The current estimate for each neuron's responsibility in the error
    std::vector<Matrix> deltas;

    // The gradient of the cost function with respect to the weights
    std::vector<Matrix> gradients;

    // The parameters provided when setting up the network
    Config m_config;

};  // ANN


/**
 * The layers are still rows, but now these rows are
 * stacked into a number of columns
 */
template<typename FT>
void ANN<FT>::setup(Config config)
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
        Matrix::Zero(input_size, BS), Matrix::Constant(1, BS, 1.0);

    for (Eigen::Index i = 0; i < n_layers - 2; ++i) {
        auto output_size = m_config.HiddenLayers[i];

        layers.emplace_back(output_size + 1, BS) <<
            Matrix::Zero(output_size, BS), Matrix::Constant(1, BS, 1.0);
        cache.emplace_back(output_size, BS) << Matrix::Zero(output_size, BS);
        deltas.emplace_back(output_size, BS) << Matrix::Zero(output_size, BS);
        weights.emplace_back(output_size, input_size + 1)
            << Matrix::Random(output_size, input_size + 1) * 2 / std::sqrt(input_size+1+output_size);  // Xavier initialization for sigmoid
        gradients.emplace_back(output_size, input_size + 1)
            << Matrix::Zero(output_size, input_size + 1);
        input_size = layout.emplace_back(output_size);
    }

    input_size = m_config.HiddenLayers[n_layers - 3];
    auto output_size = m_config.OutputSize;
    layers.emplace_back(output_size, BS) << Matrix::Zero(output_size, BS);
    deltas.emplace_back(output_size, BS) << Matrix::Zero(output_size, BS);
    cache.emplace_back(output_size, BS) << Matrix::Zero(output_size, BS);
    weights.emplace_back(output_size, input_size + 1) << Matrix::Random(output_size, input_size + 1) * 2 / std::sqrt(input_size+1+output_size);
    gradients.emplace_back(output_size, input_size + 1) << Matrix::Zero(output_size, input_size + 1);
}

template<typename FT>
template <typename Derived, typename OtherDerived>
void ANN<FT>::Forward(const Eigen::MatrixBase<Derived>& inputs,
                  Eigen::MatrixBase<OtherDerived> const& outputs_)
{
    // assign inputs to the first layer
    layers[0].topRows(inputs.rows()) = inputs;
    const Eigen::Index L = n_layers - 1;

    for (Eigen::Index i = 0; i < n_layers - 1; ++i) {
        // Cache the weighted input values
        cache[i] = weights[i] * layers[i];
        // Compute the activated neuron values
        if (i < n_layers - 2)
            activation::Sigmoid(cache[i], layers[i + 1].topRows(cache[i].rows()));
        else
            layers[i+1].topRows(cache[i].rows()) = cache[i];
    }
    // Copy the outputs in the provided matrix
    const_cast<Eigen::MatrixBase<OtherDerived>&> (outputs_) = layers[L];
}

template<typename FT>
template<typename Derived, typename OtherDerived>
void ANN<FT>::Error(const Eigen::MatrixBase<Derived>& targets,
                const Eigen::MatrixBase<OtherDerived>& predictions,
                Eigen::MatrixBase<OtherDerived> const& errors_)
{
    const_cast<Eigen::MatrixBase<OtherDerived>&> (errors_) = predictions - targets;
}

template<typename FT>
template<typename Derived>
void ANN<FT>::BackpropagateError(const Eigen::MatrixBase<Derived>& errors)
{
    deltas.back() = errors;

    for (Eigen::Index i = n_layers - 2; i > 0; --i) {
        activation::DerSigmoid(cache[i-1], cache[i-1]);
        deltas[i-1] = (weights[i].transpose() * deltas[i]).topRows(deltas[i-1].rows()).cwiseProduct(cache[i-1]);
    }
}

template<typename FT>
void ANN<FT>::CalculateGradients()
{
    for (Eigen::Index i = 0; i < n_layers - 1; ++i)
        gradients[i] = (deltas[i] * layers[i].transpose()) / static_cast<FT>(m_config.batch_size);
}

template<typename FT>
void ANN<FT>::UpdateWeights()
{
    for (Eigen::Index i = n_layers - 2; i >= 0; --i) {
        weights[i] -= (learning_rate * gradients[i]);
    }
}

template<typename FT>
template<typename Derived, typename OtherDerived>
void ANN<FT>::CostSum(const Eigen::MatrixBase<Derived>& targets, const Eigen::MatrixBase<OtherDerived>& outputs,
                  double& cost)
{
    cost += ((targets - outputs).colwise().squaredNorm()).rowwise().sum().value();
}

template<typename FT>
template<typename Derived, typename OtherDerived>
void ANN<FT>::Predict(const Eigen::MatrixBase<Derived>& inputs, Eigen::MatrixBase<OtherDerived> const& outputs_)
{
    Forward(inputs, outputs_);
}

template<typename FT>
void ANN<FT>::print(std::ostream& out) const
{
    for (int i=0; i < n_layers - 1; ++i) {
        out << "Layer " << i << "\n\n";
        out << "\nweights:\n" << weights[i];
        out << std::endl;
    }
}


}  // namespace simple

#endif  // TEMPLATEANN_H_
