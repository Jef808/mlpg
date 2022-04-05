#ifndef COSTFUNCTIONS_H_
#define COSTFUNCTIONS_H_

#include "TemplateAnn.h"
#include "Eigen/Core"


namespace simple::cost_functions {

enum class Function {
    MeanSquares,
    CrossEntropy,
    LogLikelihood
};

template<Function CostFcn>
struct AccCost {
    AccCost() = default;

    template<typename Derived, typename OtherDerived>
    void operator()(const Eigen::MatrixBase<Derived>& targets,
                    const Eigen::MatrixBase<OtherDerived>& outputs,
                    double& cost) const;
};

template<Function CostFcn>
struct OutputError {
    OutputError() = default;

    template<typename Derived, typename OtherDerived, typename OtherDerived2>
    void operator()(const Eigen::MatrixBase<Derived>& targets,
                    const Eigen::MatrixBase<OtherDerived>& outputs,
                    Eigen::MatrixBase<OtherDerived2> const& errors) const;
};

template<>
struct AccCost<Function::MeanSquares> {
    AccCost() = default;

    template<typename Derived, typename OtherDerived>
    void operator()(const Eigen::MatrixBase<Derived>& targets,
                    const Eigen::MatrixBase<OtherDerived>& outputs,
                    double& cost) const
    {
        cost += ((targets - outputs).colwise().squaredNorm()).array().sum();
    }
};

template<>
struct OutputError<Function::MeanSquares> {
    OutputError() = default;

    template<typename Derived, typename OtherDerived, typename OtherDerived2>
    void operator()(const Eigen::MatrixBase<Derived>& targets,
                    const Eigen::MatrixBase<OtherDerived>& outputs,
                    Eigen::MatrixBase<OtherDerived2> const& errors) const
    {
        const_cast< Eigen::MatrixBase<OtherDerived2>& > (errors) = outputs - targets;
    }
};

template<>
struct AccCost<Function::CrossEntropy> {
    AccCost() = default;

    template<typename Derived, typename OtherDerived>
    void operator()(const Eigen::MatrixBase<Derived>& targets,
                    const Eigen::MatrixBase<OtherDerived>& outputs,
                    double& cost) const
    {
        cost -= ((targets.array().colwise() * outputs.array.colwise().log())
            - (1.0 - targets.array().colwise()) * (1.0 - outputs.array().colwise()).log()).sum().sum();
    }
};

template<>
struct OutputError<Function::CrossEntropy> {
    OutputError() = default;

    template<typename Derived, typename OtherDerived, typename OtherDerived2>
    void operator()(const Eigen::MatrixBase<Derived>& targets,
                    const Eigen::MatrixBase<OtherDerived>& outputs,
                    Eigen::MatrixBase<OtherDerived2> const& errors) const
    {
        const_cast< Eigen::MatrixBase<OtherDerived2>& > (errors) = outputs - targets;
    }
};



}  // namespace simple::cost_functions

#endif // COSTFUNCTIONS_H_
