#ifndef FUNCTION_H
#define FUNCTION_H

#include "Eigen/Core"

#include <algorithm>


namespace simple {

/**
 * Wrap a function of one variable @F to be able to act component-wise on tensors
 */
/// TODO: Simplify this!!! I just want a wrapper around a scalar function...
template <typename FImpl>
struct RealFunction {
    RealFunction(double x_min_, double x_max_, FImpl impl_ = FImpl{})
            : impl{impl_}, x_min{x_min_}, x_max{x_max_} {}

    // This is impl itself
    double operator()(double x) const { return impl(x); }

    // Recipe to act on containers (Just works with std::vector really)
    template <template <class, class> typename F, typename T, typename Y>
    auto operator()(const F<Y, T>& in, F<Y, T>& out) const {
        return std::transform(std::begin(in), std::end(in), std::back_inserter(out),
                              [&me = impl](const Y& y) { return me(y); });
    }
    template <typename Derived1, typename Derived2>
    void operator()(const Eigen::DenseBase<Derived2>& in, Eigen::DenseBase<Derived1>& out) {
        this->impl(out.colwise().rowwise()) = this->impl(out.colwise().rowwise());
    }
    auto operator()(std::vector<double>& out) const {
        return [&](const std::vector<double>& in) { return this->operator()(in, out); };
    }

    FImpl impl;
    double x_min = -10.0;
    double x_max = 10.0;
};

auto finite_domain_lambda = []<typename F>(double xmin, double xmax, F f) {
    return RealFunction(xmin, xmax, f);
};


} // namespace

#endif /* FUNCTION_H */
