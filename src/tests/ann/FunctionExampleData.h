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


template <typename FT, typename F>
class FunctionDataCollector {
public:
    FunctionDataCollector(RealFunction<FT, F> f) : FN{f} {}

    void Training(const size_t n_train_, std::vector<FT>& train_x, std::vector<FT>& train_y) {
        lin_space(x_min(), x_max(), n_train_, std::back_inserter(train_x));
        FN(train_x, train_y);
    }

    void shuffle(std::vector<FT>& Xs, std::vector<FT>& Ys) {
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

    [[nodiscard]] FT x_min() const { return static_cast<FT>( FN.x_min ); }
    [[nodiscard]] FT x_max() const { return static_cast<FT>( FN.x_max ); }

private:
    using Buffer = std::vector<FT>;
    using backiter = std::back_insert_iterator<Buffer>;

    std::random_device rd;
    std::mt19937 eng { rd() };

    RealFunction<FT, F> FN;

    void lin_space(const FT min_x,
                   const FT max_x,
                   const size_t n_points,
                   backiter it)
    {
        FT n = x_min();
        FT step = (x_max() - x_min()) / static_cast<FT>(n_points);

        SPDLOG_DEBUG("In lin_space, x_min is {}, x_max is {}, n_points is {}, Step is {}", min_x, max_x, n_points, step);

        std::generate_n(it, n_points, [s=step, &n] () mutable {
            auto ret = n;
            n += s;
            return ret;
        });
    }

    void add_gaussian_noise(std::vector<FT>& inputs, const FT var12)
    {
        std::normal_distribution<FT> dist(0, var12);
        std::transform(inputs.begin(),
                       inputs.end(),
                       inputs.begin(), [&](auto d) { return d + dist(eng); });
    }
};


}  // namespace simple


#endif /* FUNCTIONEXAMPLEDATA_H */
