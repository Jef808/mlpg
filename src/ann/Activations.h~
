#ifndef ACTIVATIONS_H_
#define ACTIVATIONS_H_

#include "Eigen/Core"


namespace simple::activation {

template<typename FT>
FT Sigmoid_(const FT i) { return 1.0 / (1 + std::exp(-i)); }
template<typename FT>
FT DerSigmoid_(const FT i) { return Sigmoid_(i) * (1.0 - Sigmoid_(i)); }

template<typename FT>
FT Tanh_(const double i) { return std::tanh(i); }
template<typename FT>
FT DerTanh_(const FT i) { return 1.0 - std::pow(i, 2); }

template<typename Derived, typename OtherDerived>
void inline Sigmoid(const Eigen::MatrixBase<Derived>& inputs,
                    Eigen::MatrixBase<OtherDerived> const& outputs_)
{
    const_cast< Eigen::MatrixBase<OtherDerived>& >(outputs_) = inputs.array().logistic();
        //inputs.unaryExpr(std::ref(Sigmoid_<FT>));
}
template<typename Derived, typename OtherDerived, typename OtherDerived2>
void inline TimesDerSigmoid(const Eigen::MatrixBase<Derived>& activations,
                            const Eigen::MatrixBase<OtherDerived>& inputs,
                            Eigen::MatrixBase<OtherDerived2> const& outputs_)
{
    //using FT = typename OtherDerived::Scalar;
    const_cast< Eigen::MatrixBase<OtherDerived2>& >(outputs_) = (activations.array() * (1.0 - activations.array())) * inputs.array();
    //.unaryExpr(std::ref(DerSigmoid_<FT>));
}

template<typename Derived, typename OtherDerived>
void Tanh(const Eigen::MatrixBase<Derived>& inputs,
          Eigen::MatrixBase<OtherDerived> const& outputs_)
{
    const_cast< Eigen::MatrixBase<OtherDerived>& >(outputs_) = inputs.array().tanh();
}
template<typename Derived, typename OtherDerived1, typename OtherDerived2>
void TimesDerTanh(const Eigen::MatrixBase<Derived>& activs,
                  const Eigen::MatrixBase<OtherDerived1>& inputs,
                  Eigen::MatrixBase<OtherDerived2> const& outputs_)
{
    const_cast< Eigen::MatrixBase<OtherDerived2>& >(outputs_) = (1.0 - activs.array()) * (1.0 + activs.array()) * inputs.array();
}

/**
 * 0) For float numbers, any integer n > 88 satisfies e^n > MAX_FLOAT, so it doesn't take much to overflow
 * to infinity. Similarly, e^(-88) underflows to 0. But this is fine since the error is then at most
 * e^(-88)... Since SoftMax is translation invariant, we can rescale its expression
 * to avoid overflow and keep the result being a probability measure (Sum_j (sigma(z_j)) = 1).
 *
 * 1) Remove the max coefficient per column so that e^(z_i) stays between 0 and 1 (with one of them 1)
 * outputs is now [[z^(1)_j - max^(1)], ..., [z^(N)_j - max^(N)]] where each entries are
 * the normalized columns.
 *
 * 2) Now each column becomes [z^(c)_j - max^(c) - log{ Sum(z^(c)_k-max^(c)) }].
 * We don't run into issues since there is no matrix multiplication involving output in the expression
 *
 * The above can be rewritten as [ log(e^{ z^(c)_k - max^(c) } / Sum_j( e^{ z^(c)_j - max^(c) } )) ]
 * so it only remains to exponentiate it.
 */
template<typename Derived, typename OtherDerived>
void SoftMax(const Eigen::MatrixBase<Derived>& inputs,
             Eigen::MatrixBase<OtherDerived> const& outputs_)
{
    Eigen::MatrixBase< OtherDerived >& outputs = const_cast< Eigen::MatrixBase<OtherDerived>& > (outputs_);

    outputs = inputs - inputs.colwise().maxCoeff().replicate(inputs.rows(), 1);
    outputs = Eigen::exp(outputs.array() - Eigen::log(outputs.array().exp().colwise().sum()).replicate(inputs.rows(), 1));
}


template<typename Derived, typename OtherDerived, typename OtherDerived2>
void TimesDerSoftMax(const Eigen::MatrixBase<Derived>& activations,
                     const Eigen::MatrixBase<OtherDerived>& inputs,
                     Eigen::MatrixBase<OtherDerived2> const& outputs)
{
    //activations.array()
}


}  // namespace simple::activation

#endif // ACTIVATIONS_H_
