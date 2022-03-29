#ifndef ACTIVATIONS_H_
#define ACTIVATIONS_H_

#include "Eigen/Core"


double Sigmoid(const double i) { return 1.0 / (1 + std::exp(-i)); }

double DerSigmoid(const double i) { return Sigmoid(i) * (1.0 - Sigmoid(i)); }


template<typename Derived, typename DerivedOther>
void ActivationFunction(const Eigen::MatrixBase<Derived>& inputs,
                        Eigen::MatrixBase<DerivedOther> const& outputs_)
{
    const_cast< Eigen::MatrixBase<DerivedOther>& >(outputs_) = inputs.unaryExpr(std::ref(Sigmoid));
}


template<typename Derived, typename DerivedOther>
void DerActivationFunction(const Eigen::MatrixBase<Derived>& inputs,
                           Eigen::MatrixBase<DerivedOther> const& outputs_)
{
    const_cast< Eigen::MatrixBase<DerivedOther>& >(outputs_) = inputs.unaryExpr(std::ref(DerSigmoid));
}


#endif // ACTIVATIONS_H_
