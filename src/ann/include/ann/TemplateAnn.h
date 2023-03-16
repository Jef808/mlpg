#ifndef TEMPLATEANN_H_
#define TEMPLATEANN_H_

#include "ann/Activations.h"
#include "ann/Config.h"
#include "ann/Shuffle_data.h"

#include "ann/utilities.h"

#include "Eigen/Core"

#include <cstddef>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>

namespace simple {

template<typename FT> class ANN
{
public:
  using Matrix = typename std::conditional<std::is_same_v<FT, float>, Eigen::MatrixXf, Eigen::MatrixXd>::type;
  using Array = typename std::conditional<std::is_same_v<FT, float>, Eigen::ArrayXf, Eigen::ArrayXd>::type;

  ANN() = default;

  /**
   * Constructs the data-structure of the model before
   * it can be used for training.
   */
  void setup(Config config);

  /**
   * Run a Forward and Backward pass for each batch of data. Update
   * gradients after each batch.
   *
   * Barebone implementation:
   * invoke the notify callback \p exit_cb after posting a batch. The consumer
   * can then count the notifications and always know how much fresh data
   * it is allowed to read.
   *
   * \param loss_out optional buffer to output \e real-time feedback on the evolution
   *        of the model.
   * \param accuracy_out Similar to \p loss_out but to output accuracy of predictions.
   * \param exit_cb Callback called at the end of each batch processed.
   */
  template<typename DataCb = NullFunctor, typename OnExitCb = NullFunctor>
  void Train(size_t NEpochs,
    std::remove_reference_t<FT> *train_x,
    std::remove_reference_t<FT> *train_y,
    ptrdiff_t n_batch,
    DataCb loss_out = DataCb{},
    DataCb accuracy_out = DataCb{},
    OnExitCb exit_cb = OnExitCb{});

  /**
   * Compute the errors with the current model with respect to the given test
   * data \e test_x provided via \p inputs and \e test_y provided via \p outputs.
   */
  template<typename Derived, typename OtherDerived>
  void Predict(const Eigen::MatrixBase<Derived> &inputs, Eigen::MatrixBase<OtherDerived> const &outputs);

  /**
   * Feed \p inputs into the network and collect the \p outputs at the last layer.
   *
   * \param inputs The initial inputs fed to the network.
   * \param outputs The <em>return value</em> of the network.
   */
  template<typename Derived, typename OtherDerived>
  void Forward(const Eigen::MatrixBase<Derived> &inputs, Eigen::MatrixBase<OtherDerived> const &outputs);

  /**
   * Runs a backpropagation pass, upgrading the weights
   * from knowledge of the last forward pass.
   *
   * \target is the expected output coming from the data,
   * \gradients output the gradient \f$nabla_W C\f$ of the cost function
   * viewed as having fixed input and output, with variable weights
   */
  void Backward();

  /**
   * Given the training targets, get the sum of the loss function
   * for a given batch of outputs.
   *
   * \param targets The target values from the dataset, batched one per column.
   * \param outputs The outputs predicted by the current model, batched one per column.
   * \param double The sum of the cost function for each pair of columns.
   */
  template<typename Derived, typename OtherDerived>
  void AccLoss(const Eigen::MatrixBase<Derived> &targets, const Eigen::MatrixBase<OtherDerived> &outputs, double &cost);

  template<typename Derived, typename OtherDerived>
  void AccNbSuccess(const Eigen::MatrixBase<Derived> &targets,
    const Eigen::MatrixBase<OtherDerived> &outputs,
    int &n_success);

  /**
   * Calculate the Output layer's deltas in terms of Expected/Obtained outputs
   *
   * \param targets The target values from training dataset, batched one per column.
   * \param outputs The outputs predicted by the current model, batched one per column.
   * \param errors  Output matrix where we store the resulting errors.
   */
  template<typename Derived, typename OtherDerived>
  void Error(const Eigen::MatrixBase<Derived> &targets,
    const Eigen::MatrixBase<OtherDerived> &outputs,
    Eigen::MatrixBase<OtherDerived> const &errors);

  /**
   * Propagate the error backwards and compute the gradients of the cost function.
   *
   * For each layer, compute the deltas, i.e. the \f$\partial (a^L) / \partial z^l\f$,
   * when \f$C\f$ is seen as having fixed weights and output, with variable input
   *
   * \param errors The errors computed from observation at the output layer
   * \param gradients The gradients of the Cost function with respect to the current weights
   *
   */
  template<typename Derived> void BackpropagateError(const Eigen::MatrixBase<Derived> &errors);

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
  void print(std::ostream &out) const;

  [[nodiscard]] const std::vector<Eigen::Index> &get_layout() const;

  [[nodiscard]] std::vector<const FT *> get_weights() const;

  [[nodiscard]] std::vector<const FT *> get_activations() const;

private:
  // The number of layers in the network
  Eigen::Index n_layers{ 0 };

  // The number of inputs per input nodes
  Eigen::Index n_batches{ 0 };

  // Controls the step size when updating the network's weights
  FT learning_rate;

  // The current shape of the network, as set by the configuration
  std::vector<Eigen::Index> layout;

  // The current weights of the network
  std::vector<Matrix> weights;

  // The current value of the network's activations
  std::vector<Matrix> layers;

  // To save the weighted inputs during a forward pass
  // std::vector<Matrix> cache;

  // The current estimate for each neuron's responsibility in the error
  std::vector<Matrix> deltas;

  // The gradient of the cost function with respect to the weights
  std::vector<Matrix> gradients;

  // The parameters provided when setting up the network
  Config m_config;

};// ANN


/**
 * The layers are still rows, but now these rows are
 * stacked into a number of columns
 */
template<typename FT> void ANN<FT>::setup(Config config)
{
  m_config = std::move(config);

  // Clear the state
  layout.clear();
  layers.clear();
  // cache.clear();
  deltas.clear();
  weights.clear();
  gradients.clear();

  std::random_device rnd;
  std::mt19937 eng{ rnd() };
  std::normal_distribution<FT> dist(0, 1.0);
  std::vector<FT> rand_init;

  Eigen::Index bsize = m_config.batch_size;

  n_layers = static_cast<Eigen::Index>(m_config.HiddenLayers.size()) + 2;
  learning_rate = m_config.LearningRate;

  auto input_size = layout.emplace_back(m_config.InputSize);
  layers.emplace_back(input_size + 1, bsize) << Matrix::Zero(input_size, bsize), Matrix::Constant(1, bsize, 1.0);

  for (Eigen::Index i = 0; i < n_layers - 2; ++i) {
    auto output_size = m_config.HiddenLayers[i];

    std::generate_n(std::back_inserter(rand_init),
      output_size * (input_size + 1),
      [&dist, &eng, scale = 2 / std::sqrt(input_size + 1)]() { return dist(eng) * scale; });

    Matrix rand = Eigen::Map<Matrix>(rand_init.data(), output_size, input_size + 1);

    layers.emplace_back(output_size + 1, bsize) << Matrix::Zero(output_size, bsize), Matrix::Constant(1, bsize, 1.0);
    // cache.emplace_back(output_size, BS) << Matrix::Zero(output_size, BS);
    deltas.emplace_back(output_size, bsize) << Matrix::Zero(output_size, bsize);
    weights.emplace_back(output_size, input_size + 1) << rand;
    //<< Matrix::Random(output_size, input_size + 1) * (2 / std::sqrt(input_size));  // Xavier initialization for
    //sigmoid
    gradients.emplace_back(output_size, input_size + 1) << Matrix::Zero(output_size, input_size + 1);
    input_size = layout.emplace_back(output_size);
  }

  input_size = m_config.HiddenLayers[n_layers - 3];
  auto output_size = layout.emplace_back(m_config.OutputSize);

  std::generate_n(std::back_inserter(rand_init),
    output_size * (input_size + 1),
    [&dist, &eng, scale = 2 / std::sqrt(input_size + 1)]() { return dist(eng) * scale; });

  Matrix rand = Eigen::Map<Matrix>(rand_init.data(), output_size, input_size + 1);

  layers.emplace_back(output_size, bsize) << Matrix::Zero(output_size, bsize);
  deltas.emplace_back(output_size, bsize) << Matrix::Zero(output_size, bsize);
  // cache.emplace_back(output_size, BS) << Matrix::Zero(output_size, bsize);
  weights.emplace_back(output_size, input_size + 1)
    << rand;// Matrix::Random(output_size, input_size + 1) * (2 / std::sqrt(input_size));
  gradients.emplace_back(output_size, input_size + 1) << Matrix::Zero(output_size, input_size + 1);
}

template<typename FT>
template<typename Derived, typename OtherDerived>
void ANN<FT>::Forward(const Eigen::MatrixBase<Derived> &inputs, Eigen::MatrixBase<OtherDerived> const &outputs)
{
  // assign inputs to the first layer
  layers[0].topRows(inputs.rows()) = inputs;
  // the 0-based index of the last layer
  const Eigen::Index layers_end = n_layers - 1;

  // Apply activation in the hidden layers
  for (Eigen::Index i = 0; i < layers_end - 1; ++i) {

    activation::Sigmoid(weights[i] * layers[i], layers[i + 1].topRows(layout[i + 1]));
    // Compute the activated neuron values
  }
  // Linear output layer
  // const_cast< Eigen::MatrixBase<OtherDerived>& >(outputs_) = weights[L-1] * layers[L-1];

  // activation::Sigmoid(
  //     weights[L-1] * layers[L-1],
  //     const_cast< Eigen::MatrixBase<OtherDerived>& >(outputs_));
  // layers[L]

  // Softmax output layer
  activation::SoftMax(
    weights[layers_end - 1] * layers[layers_end - 1], const_cast<Eigen::MatrixBase<OtherDerived> &>(outputs));
}

template<typename FT>
template<typename Derived, typename OtherDerived>
void ANN<FT>::Error(const Eigen::MatrixBase<Derived> &targets,
  const Eigen::MatrixBase<OtherDerived> &outputs,
  Eigen::MatrixBase<OtherDerived> const &errors)
{
  const_cast<Eigen::MatrixBase<OtherDerived> &>(errors) = outputs - targets;
}

template<typename FT>
template<typename Derived>
void ANN<FT>::BackpropagateError(const Eigen::MatrixBase<Derived> &errors)
{
  deltas.back() = errors;

  for (Eigen::Index i = n_layers - 2; i > 0; --i) {
    activation::TimesDerSigmoid(
      layers[i].topRows(layout[i]), (weights[i].transpose() * deltas[i]).topRows(layout[i]), deltas[i - 1]);
  }
}

template<typename FT> void ANN<FT>::CalculateGradients()
{
  for (Eigen::Index i = n_layers - 2; i >= 0; --i) { gradients[i] = (deltas[i] * layers[i].transpose()); }
}

template<typename FT> void ANN<FT>::UpdateWeights()
{
  for (Eigen::Index i = n_layers - 2; i >= 0; --i) {
    // The regularization contribution
    weights[i].leftCols(layout[i]) *=
      (1.0 - (learning_rate * m_config.L2RegCoeff / std::ceil(m_config.n_data * m_config.training_ratio)));

    // Gradient contribution
    weights[i] -= (learning_rate * gradients[i]) / static_cast<FT>(m_config.batch_size);
  }
}

template<typename FT>
template<typename DataCb, typename OnExitCb>
void ANN<FT>::Train(size_t NEpochs,
  std::remove_reference_t<FT> *train_x,
  std::remove_reference_t<FT> *train_y,
  ptrdiff_t n_batch,
  DataCb avg_loss_out,
  DataCb avg_accuracy_out,
  OnExitCb exit_cb)
{
  Data::Shuffler Shuffle;

  const auto bsize = m_config.batch_size;
  // const auto data_size = n_batch * bsize;

  Matrix outputs(m_config.OutputSize, bsize);
  Matrix errors(m_config.OutputSize, bsize);

  for (auto epoch = 0; epoch < NEpochs; ++epoch) {
    double loss = 0.0;
    // int n_success = 0;

    // Shuffle the batches before beginning training.
    Shuffle.shuffle(
      train_x, train_y, train_x + bsize * m_config.InputSize, train_y + bsize * m_config.OutputSize, n_batch);

    for (auto i = 0; i < n_batch; ++i) {
      double batch_loss = 0.0;
      auto batch_n_success = 0;

      // Note: This constructor is the dynamic-size matrix case.
      Eigen::Map<Matrix> batch_x(train_x + i * bsize * m_config.InputSize,// dataPtr
        m_config.InputSize,// rows
        bsize);// cols
      Eigen::Map<Matrix> batch_y(train_y + i * bsize * m_config.OutputSize, m_config.OutputSize, bsize);

      Forward(batch_x, outputs);

      AccLoss(outputs, batch_y, batch_loss);
      // AccNbSuccess(outputs, batch_y, batch_n_success);

      loss += batch_loss;
      // n_success += batch_n_success;

      if constexpr (not std::is_same_v<DataCb, std::nullptr_t>) {
        avg_loss_out(batch_loss / bsize);
        avg_accuracy_out(static_cast<float>(batch_n_success) / bsize);
      }

      Error(batch_y, outputs, errors);

      BackpropagateError(errors);

      CalculateGradients();

      UpdateWeights();
    }

    std::cout << "Epoch " << epoch << "... Average loss: " << loss / (n_batch * bsize) << std::endl;
    ;
  }

  if constexpr (not std::is_same_v<OnExitCb, std::nullptr_t>) { exit_cb.notify(); }
}

template<typename FT>
template<typename Derived, typename OtherDerived>
void ANN<FT>::AccLoss(const Eigen::MatrixBase<Derived> &targets,
  const Eigen::MatrixBase<OtherDerived> &outputs,
  double &loss)
{
  // Mean Squared Differences
  // loss += ((targets - outputs).colwise().squaredNorm()).array().sum();

  // Log-Likelihood
  loss -= Eigen::log((targets.array() * targets.array()).colwise().maxCoeff()).sum();

  // Cross Entropy
  // loss -= (targets.array() * outputs.array().log() + (1.0 - targets.array()) * (1.0 -
  // outputs.array()).log()).colwise().sum().sum();
}

template<typename FT>
template<typename Derived, typename OtherDerived>
void ANN<FT>::AccNbSuccess(const Eigen::MatrixBase<Derived> &targets,
  const Eigen::MatrixBase<OtherDerived> &outputs,
  int &n_success)
{
  assert(outputs.rows() == targets.rows() && outputs.rows() == targets.rows()
         && "Size mismatch between outputs and predictions");

  for (Eigen::Index idx = 0; idx < targets.cols(); ++idx) {
    // Store get the index of the maximum coefficient of each column outputs
    int node = -1;
    targets.col(idx).maxCoeff(&node);

    assert(node >= 0 && "Failed to record index of choice");
    assert(node < targets.cols() && "index of choice is bigger than number of classes");

    if (outputs(node, idx) > 0.8f) { ++n_success; }
  }
}

template<typename FT>
template<typename Derived, typename OtherDerived>
void ANN<FT>::Predict(const Eigen::MatrixBase<Derived> &inputs, Eigen::MatrixBase<OtherDerived> const &outputs)
{
  Forward(inputs, outputs);
}

template<typename FT> void ANN<FT>::print(std::ostream &out) const
{
  for (int i = 0; i < n_layers - 1; ++i) {
    out << "Layer " << i << "\n\n";
    out << "\nweights:\n" << weights[i];
    out << std::endl;
  }
}


template<typename FT> const std::vector<Eigen::Index> &ANN<FT>::get_layout() const { return layout; }

template<typename FT> std::vector<const FT *> ANN<FT>::get_weights() const
{
  std::vector<const FT *> ret;
  std::transform(
    weights.begin(), weights.end(), std::back_inserter(ret), [](const auto &weight_mat) { return weight_mat.data(); });
  return ret;
}


template<typename FT> std::vector<const FT *> ANN<FT>::get_activations() const
{
  std::vector<const FT *> ret;
  std::transform(layers.begin(), layers.end(), std::back_inserter(ret), [](const auto &layer) { return layer.data(); });
  return ret;
}

}// namespace simple

#endif// TEMPLATEANN_H_
