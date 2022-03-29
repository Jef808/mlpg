#ifndef FUNCTIONEXAMPLEDATA_H
#define FUNCTIONEXAMPLEDATA_H

#include "Function.h"

#include "spdlog/spdlog.h"

#include <algorithm>
#include <iostream>
#include <random>
#include <vector>
#include <utility>
#include <iterator>

namespace simple {


template <typename F>
class FunctionDataCollector {
public:
    FunctionDataCollector(RealFunction<F> f) : FN{f} {}

    void Training(const size_t n_train_, std::vector<double>& train_x, std::vector<double>& train_y) {
        lin_space(x_min(), x_max(), n_train_, std::back_inserter(train_x));
        FN(train_x, train_y);
    }

    void shuffle(std::vector<double>& Xs, std::vector<double>& Ys) {
        using diff_t = std::vector<double>::difference_type;
        using distr_t = std::uniform_int_distribution<diff_t>;
        using param_t = distr_t::param_type;

        distr_t D;
        diff_t n = Xs.size();
        for (diff_t i = n-1; i > 0; --i) {
            using std::swap;
            auto res = D(eng, param_t(0, i));
            swap(Xs[i], Xs[res]);
            swap(Ys[i], Ys[res]);
        }
    }

    [[nodiscard]] double x_min() const { return FN.x_min; }
    [[nodiscard]] double x_max() const { return FN.x_max; }

private:
    using Buffer = std::vector<double>;
    using iter = std::vector<double>::iterator;
    using backiter = std::back_insert_iterator<Buffer>;

    std::random_device rd;
    std::mt19937 eng { rd() };

    RealFunction<F> FN;

    void lin_space(const double min_x,
                   const double max_x,
                   const size_t n_points,
                   backiter it)
    {
        double step = (max_x - min_x) / static_cast<double>(n_points);
        SPDLOG_DEBUG("In lin_space, x_min is {}, x_max is {}, n_points is {}, Step is {}", min_x, max_x, n_points, step);

        auto n = static_cast<double>(min_x);
        std::generate_n(it, n_points,
                        [s=step, &n] () mutable {
                            auto ret = n;
                            n += s;
                            return ret;
                        });
    }

    void add_gaussian_noise(std::vector<double>& inputs, const double var12)
    {
        std::normal_distribution<> dist(0, var12);
        std::transform(inputs.begin(),
                       inputs.end(),
                       inputs.begin(), [&](auto d) { return d + dist(eng); });
    }
};


}  // namespace simple


#endif /* FUNCTIONEXAMPLEDATA_H */
