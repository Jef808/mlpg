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

template<typename Derived, typename DerivedOther>
void Sigmoid(const Eigen::MatrixBase<Derived>& inputs,
                        Eigen::MatrixBase<DerivedOther> const& outputs_)
{
    using FT = typename DerivedOther::Scalar;
    const_cast< Eigen::MatrixBase<DerivedOther>& >(outputs_) = inputs.unaryExpr(std::ref(Sigmoid_<FT>));
}
template<typename Derived, typename DerivedOther>
void DerSigmoid(const Eigen::MatrixBase<Derived>& inputs,
                           Eigen::MatrixBase<DerivedOther> const& outputs_)
{
    using FT = typename DerivedOther::Scalar;
    const_cast< Eigen::MatrixBase<DerivedOther>& >(outputs_) = inputs.unaryExpr(std::ref(DerSigmoid_<FT>));
}

template<typename Derived, typename DerivedOther>
void Tanh(const Eigen::MatrixBase<Derived>& inputs,
                        Eigen::MatrixBase<DerivedOther> const& outputs_)
{
    using FT = typename DerivedOther::Scalar;
    const_cast< Eigen::MatrixBase<DerivedOther>& >(outputs_) = inputs.unaryExpr(std::ref(Tanh_<FT>));
}
template<typename Derived, typename DerivedOther>
void DerTanh(const Eigen::MatrixBase<Derived>& inputs,
                           Eigen::MatrixBase<DerivedOther> const& outputs_)
{
    using FT = typename DerivedOther::Scalar;
    const_cast< Eigen::MatrixBase<DerivedOther>& >(outputs_) = inputs.unaryExpr(std::ref(DerTanh_<FT>));
}

}  // namespace simple::activation

#endif // ACTIVATIONS_H_
