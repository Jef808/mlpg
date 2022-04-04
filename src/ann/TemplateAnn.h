#ifndef TEMPLATEANN_H_
#define TEMPLATEANN_H_

#include "Activations.h"
#include "Config.h"
#include "data/manip.h"

#include "Eigen/Core"

#include <cstddef>
#include <iostream>
#include <random>
#include <vector>
#include <type_traits>

namespace simple {

template<typename FT>
class ANN {
public:
    using Matrix = typename std::conditional< std::is_same_v<FT, float>,
                                              Eigen::MatrixXf, Eigen::MatrixXd >::type;
    using Array = typename std::conditional< std::is_same_v<FT, float>,
                                              Eigen::ArrayXf, Eigen::ArrayXd >::type;

    ANN() = default;

    /**
     * Constructs the data-structure of the model before
     * it can be used for training.
     */
    void setup(Config);

    /**
     * Run a Forward and Backward pass for each batch of data. Update
     * gradients after each batch.
     *
     * @losses and @predictions are optional buffers to output "real-time"
     * feedback on how the model is evolving. Barebone implementation:
     * invoke the notify callback @CB after posting a batch. The consumer
     * can then count the notifications and always know how much fresh data
     * it is allowed to read. (Consumer owns the memory pool)
     */
    template<typename DataCb = std::nullptr_t, typename OnExitCb = std::nullptr_t>
    void Train(size_t NEpochs, FT* train_x, FT* train_y, ptrdiff_t n_training,
               DataCb loss_out=nullptr, DataCb accuracy_out=nullptr,
               OnExitCb exit_cb=nullptr);

    /**
     * Compute the errors with the current model with respect to the given test
     * data @test_x and @test_y.
     */
    template<typename Derived, typename OtherDerived>
    void Predict(const Eigen::MatrixBase<Derived>& inputs,
                 Eigen::MatrixBase<OtherDerived> const& predictions);

    template< typename Derived, typename OtherDerived >
    void AccSuccess(const Eigen::MatrixBase<Derived>& outputs,
                  const Eigen::MatrixBase<OtherDerived>& targets,
                  unsigned int& success);

    /**
     * Feed @inputs into the network and collect the @outputs at the last layer.
     */
    template <typename Derived, typename OtherDerived>
    void Forward(const Eigen::MatrixBase<Derived>& inputs,
                 Eigen::MatrixBase<OtherDerived> const& outputs);

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
    void AccLoss(const Eigen::MatrixBase<Derived>& targets,
                 const Eigen::MatrixBase<OtherDerived>& outputs,
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

    [[nodiscard]] const std::vector<Eigen::Index>& get_layout() const;

    [[nodiscard]] std::vector<FT*> get_weights();

    [[nodiscard]] std::vector<FT*> get_activations();

private:
    // The number of layers in the network
    Eigen::Index n_layers;

    // The number of inputs per input nodes
    Eigen::Index n_batches;

    // Controls the step size when updating the network's weights
    FT learning_rate;

    // The current shape of the network, as set by the configuration
    std::vector<Eigen::Index> layout;

    // The current weights of the network
    std::vector<Matrix> weights;

    // The current value of the network's activations
    std::vector<Matrix> layers;

    // To save the weighted inputs during a forward pass
    //std::vector<Matrix> cache;

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
    //cache.clear();
    deltas.clear();
    weights.clear();
    gradients.clear();

    std::random_device rd;
    std::mt19937 eng{ rd() };
    std::normal_distribution<float> dist(0, 1.0);
    std::vector<float> rand_init;

    Eigen::Index BS = m_config.batch_size;

    n_layers = static_cast<Eigen::Index>(m_config.HiddenLayers.size()) + 2;
    learning_rate = m_config.LearningRate;

    auto input_size = layout.emplace_back(m_config.InputSize);
    layers.emplace_back(input_size + 1, BS) <<
        Matrix::Zero(input_size, BS), Matrix::Constant(1, BS, 1.0);

    for (Eigen::Index i = 0; i < n_layers - 2; ++i) {
        auto output_size = m_config.HiddenLayers[i];

        std::generate_n(std::back_inserter(rand_init), output_size * (input_size + 1),
                        [&dist, &eng, scale=2/std::sqrt(input_size+1)](){ return dist(eng) * scale; });

        Matrix rand = Eigen::Map<Matrix>(rand_init.data(), output_size, input_size + 1);

        layers.emplace_back(output_size + 1, BS) <<
            Matrix::Zero(output_size, BS), Matrix::Constant(1, BS, 1.0);
        //cache.emplace_back(output_size, BS) << Matrix::Zero(output_size, BS);
        deltas.emplace_back(output_size, BS) << Matrix::Zero(output_size, BS);
        weights.emplace_back(output_size, input_size + 1) << rand;
            //<< Matrix::Random(output_size, input_size + 1) * (2 / std::sqrt(input_size));  // Xavier initialization for sigmoid
        gradients.emplace_back(output_size, input_size + 1)
            << Matrix::Zero(output_size, input_size + 1);
        input_size = layout.emplace_back(output_size);
    }

    input_size = m_config.HiddenLayers[n_layers - 3];
    auto output_size = m_config.OutputSize;

    std::generate_n(std::back_inserter(rand_init), output_size * (input_size + 1),
                        [&dist, &eng, scale=2/std::sqrt(input_size+1)](){ return dist(eng) * scale; });

    Matrix rand = Eigen::Map<Matrix>(rand_init.data(), output_size, input_size + 1);

    layers.emplace_back(output_size, BS) << Matrix::Zero(output_size, BS);
    deltas.emplace_back(output_size, BS) << Matrix::Zero(output_size, BS);
    //cache.emplace_back(output_size, BS) << Matrix::Zero(output_size, BS);
    weights.emplace_back(output_size, input_size + 1) << rand;//Matrix::Random(output_size, input_size + 1) * (2 / std::sqrt(input_size));
    gradients.emplace_back(output_size, input_size + 1) << Matrix::Zero(output_size, input_size + 1);
}

template<typename FT>
template <typename Derived, typename OtherDerived>
void ANN<FT>::Forward(const Eigen::MatrixBase<Derived>& inputs,
                      Eigen::MatrixBase<OtherDerived> const& outputs_)
{
    // assign inputs to the first layer
    layers[0].topRows(inputs.rows()) = inputs;
    // the 0-based index of the last layer
    const Eigen::Index L = n_layers - 1;

    // Apply activation in the hidden layers
    for (Eigen::Index i = 0; i < L - 1; ++i) {

        activation::Sigmoid(weights[i] * layers[i], layers[i+1].topRows(layout[i+1]));
        // Compute the activated neuron values
    }
    // Linear output layer
    //const_cast< Eigen::MatrixBase<OtherDerived>& >(outputs_) = weights[L-1] * layers[L-1];

    // activation::Sigmoid(
    //     weights[L-1] * layers[L-1],
    //     const_cast< Eigen::MatrixBase<OtherDerived>& >(outputs_));
    // layers[L]

    // Softmax output layer
    activation::SoftMax(
        weights[L-1] * layers[L-1],
        const_cast< Eigen::MatrixBase<OtherDerived>& >(outputs_));
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
        activation::TimesDerSigmoid(layers[i].topRows(layout[i]),
                                    (weights[i].transpose() * deltas[i]).topRows(layout[i]),
                                    deltas[i-1]);
    }
}

template<typename FT>
void ANN<FT>::CalculateGradients()
{
    for (Eigen::Index i = n_layers - 2; i >= 0; --i)
        gradients[i] = (deltas[i] * layers[i].transpose());
}

template<typename FT>
void ANN<FT>::UpdateWeights()
{
    for (Eigen::Index i = n_layers - 2; i >= 0; --i) {
        // The regularization contribution
        weights[i].leftCols(layout[i]) *= (1.0 - (learning_rate * m_config.L2RegCoeff / std::ceil(m_config.n_data * m_config.training_ratio)));

        // Gradient contribution
        weights[i] -= (learning_rate * gradients[i]) / static_cast<FT>(m_config.batch_size);
    }
}

template<typename FT>
template< typename DataCb, typename OnExitCb >
void ANN<FT>::Train(const size_t NEpochs, FT* train_x, FT* train_y, ptrdiff_t n_batch,
                    DataCb avg_loss_out, DataCb avg_accuracy_out,
                    OnExitCb exit_cb)
{
    Data::Shuffler DS;

    const auto batch_size = m_config.batch_size;
    const auto data_size = n_batch * batch_size;

    Matrix outputs(m_config.OutputSize,
                   batch_size);
    Matrix errors(m_config.OutputSize,
                  batch_size);

    for (ptrdiff_t epoch = 0; epoch < NEpochs; ++epoch)
    {
        double loss = 0.0;
        unsigned int n_success = 0;

        DS.shuffle(train_x,
                   train_y,
                   train_x + data_size * m_config.InputSize,
                   train_y + data_size * m_config.OutputSize,
                   data_size);

        for (ptrdiff_t i = 0; i < n_batch; ++i)
        {
            Eigen::Map<Matrix> batch_x(train_x + i * batch_size * m_config.InputSize,
                                       m_config.InputSize,
                                       batch_size);
            Eigen::Map<Matrix> batch_y(train_y + i * batch_size * m_config.OutputSize,
                                       m_config.OutputSize,
                                       batch_size);

            Forward(batch_x, outputs);

            double batch_loss = 0.0;
            unsigned int batch_n_success = 0;
            AccLoss(outputs, batch_y, batch_loss);
            AccSuccess(outputs, batch_y, batch_n_success);

            loss += batch_loss;
            n_success += batch_n_success;

            if constexpr(not std::is_same_v<DataCb, std::nullptr_t>)
            {
                avg_loss_out(batch_loss / batch_size);
                avg_accuracy_out(static_cast<float>(batch_n_success) / batch_size);
            }

            Error(batch_y, outputs, errors);

            BackpropagateError(errors);

            CalculateGradients();

            UpdateWeights();
        }

        std::cout << "Epoch "<< epoch
                  << "... Average loss: "
                  << loss / (n_batch * batch_size) << std::endl;;
    }

    if constexpr (not std::is_same_v< OnExitCb, std::nullptr_t >) {
        exit_cb.notify();
    }
}

template<typename FT>
template<typename Derived, typename OtherDerived>
void ANN<FT>::AccLoss(const Eigen::MatrixBase<Derived>& targets, const Eigen::MatrixBase<OtherDerived>& outputs,
                      double& loss)
{
    // Mean Squared Differences
    //loss += ((targets - outputs).colwise().squaredNorm()).array().sum();

    // Log-Likelihood
    loss -= Eigen::log((targets.array() * outputs.array()).colwise().maxCoeff()).sum();

    // Cross Entropy
    //loss -= (targets.array() * outputs.array().log() + (1.0 - targets.array()) * (1.0 - outputs.array()).log()).colwise().sum().sum();
}

template<typename FT>
template<typename Derived, typename OtherDerived>
void ANN<FT>::AccSuccess(const Eigen::MatrixBase<Derived>& outputs,
                         const Eigen::MatrixBase<OtherDerived>& targets,
                         unsigned int& n_successful)
{
    assert(outputs.rows() == targets.rows()
           && outputs.rows() == targets.rows()
           && "Size mismatch between outputs and predictions");

    for (Eigen::Index x = 0; x < outputs.cols(); ++x) {
        // Store get the index of the maximum coefficient of each column outputs
        int y = -1;
        outputs.col(x).maxCoeff( &y );

        assert(y >= 0 && "Failed to record index of choice");
        assert(y < targets.cols() && "index of choice is bigger than number of classes");

        if (targets(y, x) > 0.8f)
            ++n_successful;
    }
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


template<typename FT>
const std::vector<Eigen::Index>& ANN<FT>::get_layout() const
{
    return layout;
}

template<typename FT>
std::vector<FT*> ANN<FT>::get_weights()
{
    std::vector<FT*> ret;
    for (auto& weight : weights) {
        ret.push_back(weight.data());
    }
    return ret;
}


template<typename FT>
std::vector<FT*> ANN<FT>::get_activations()
{
    std::vector<FT*> ret;
    for (auto& layer : layers) {
        ret.push_back(layer.data());
    }
    return ret;
}

}  // namespace simple

#endif  // TEMPLATEANN_H_
