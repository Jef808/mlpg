#ifndef FUNCTION_H
#define FUNCTION_H

#include "Eigen/Dense"

#include <algorithm>


namespace simple {

/**
 * Wrap a function of one variable @F to be able to act component-wise on tensors
 */
/// TODO: Simplify this!!! I just want a wrapper around a scalar function...
template <typename FT, typename FImpl>
struct RealFunction {
    RealFunction(FT x_min_, FT x_max_, FImpl impl_ = FImpl{})
            : impl{impl_}, x_min{x_min_}, x_max{x_max_} {}

    // This is impl itself
    FT operator()(FT x) const { return impl(x); }

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
    auto operator()(const std::vector<FT>& in, std::vector<FT>& out) const {
        std::transform(in.begin(), in.end(), std::back_inserter(out),
                       [&](auto x) { return impl(x); });
    }

    FImpl impl;
    FT x_min = -10.0;
    FT x_max = 10.0;
};

template<typename FT=double>
auto finite_domain_lambda = []<typename F>(FT xmin, FT xmax, F f) {
    return RealFunction(xmin, xmax, f);
};


} // namespace

#endif /* FUNCTION_H */
