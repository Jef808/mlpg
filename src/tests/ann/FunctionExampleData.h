#ifndef FUNCTIONEXAMPLEDATA_H
#define FUNCTIONEXAMPLEDATA_H

#include "Function.h"
#include "Eigen/Core"

#include <iostream>
#include <random>
#include <vector>
#include <utility>

namespace simple {


template <typename F>
class DataCollector {
public:
    DataCollector(RealFunction<F> f) : FN{f} {}

    std::pair<double*, double*> pData_Training() {
        return std::make_pair(buf_trainx.data(), buf_trainy.data());
    }

    // TODO Check that data is still there
    std::pair<double*, double*> pData_Testing() { return std::make_pair(buf_testx.data(), buf_testy.data()); }

    std::pair<double*, double*> pData_Sample(const size_t N, const double var12) {
        generate_sample(N, var12);
        return std::make_pair(buf_samplex.data(), buf_sampley.data());
    }

    void InitTrainingData(const size_t n_train_) {
        this->n_train = static_cast<Eigen::Index>(n_train);

        fill_domain_random(buf_trainx, n_train_);
        FN(buf_trainy, buf_trainx);
    }

    void ResetTestData(const size_t n_tests, float min_allowed_error = 0.0001f) {
        fill_domain_random(buf_testx, n_tests);
        FN(buf_testy, buf_testx);

        bool okay = test_data(buf_trainx, buf_trainy, min_allowed_error);
        assert(okay && "ERROR in DataCollector::InitTrainingData");
    }

    [[nodiscard]] double x_min() const { return FN.xmin; }
    [[nodiscard]] double x_max() const { return FN.xmax; }

private:
    std::random_device rd;
    std::mt19937 eng;

    RealFunction<F> FN;
    Eigen::Index n_train;
    float epsilon_precision{0.00001f};
    bool initialized;
    std::vector<double> buf_trainx;
    std::vector<double> buf_trainy;
    std::vector<double> buf_samplex;
    std::vector<double> buf_sampley;
    std::vector<double> buf_testx;
    std::vector<double> buf_testy;


    std::vector<double>& fill_domain_random(std::vector<double>& outputs, const size_t N) {
        outputs.clear();

        std::uniform_real_distribution<> dist(FN.x_min, FN.x_max);
        std::generate_n(std::back_inserter(outputs), N, [&] { return dist(eng); });
        return outputs;
    }

    double add_gaussian_noise(double a, double var12 = 1.0) {
        std::normal_distribution<> dist{0, var12};
        return a + dist(eng);
    }

    template <typename Fun>
    auto add_gaussian_noise(Fun fun, double var12 = 1.0) {
        std::normal_distribution<> dist{0, var12};
        return [f = fun, n = dist(eng)](const auto& w) { return n + f(w); };
    }

    void add_gaussian_noise(std::vector<double>& inputs, const double var12) {
        std::normal_distribution<> dist(0, var12);
        std::transform(inputs.begin(), inputs.end(), inputs.begin(), [&](auto d) { return d + dist(eng); });
    }

    void generate_sample(const size_t N, double var12 = 1.0) {
        buf_samplex.clear();
        buf_sampley.clear();
        std::uniform_int_distribution<> dist(0, buf_trainx.size() - 1);
        std::generate_n(std::back_inserter(buf_samplex), N, [&] { return buf_trainx[dist(eng)]; });

        FN(buf_sampley, buf_samplex);
        add_gaussian_noise(buf_sampley, var12);
    }

    bool test_data(const std::vector<double>& X, const std::vector<double>& Y, float min_allowed_error) const {
        assert((not X.empty()) && X.size() == Y.size() && "ERROR in DataCollector::test_data()");
        bool ret = true;
        double errs = 0.0;
        for (auto i = 0; i < n_train; ++i) {
            auto err = Y[i] - std::cos(X[i]);
            if (err > min_allowed_error) {
                std::cout << "Warning: the " << i << "'th pair had an error of " << err << std::endl;
                ret = false;
            }
        }
        return ret;
    }
};


}  // namespace simple


#endif /* FUNCTIONEXAMPLEDATA_H */
