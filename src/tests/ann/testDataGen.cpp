#include "FunctionExampleData.h"
#include "Function.h"

#include <cmath>

using namespace simple;


int main(int argc, char *argv[]) {
    auto function = RealFunction{-6.0, 6.0, [](auto x) { return std::cos(x); }};
    FunctionDataCollector DC{ function };

    std::vector<double> train_x;
    std::vector<double> train_y;

    DC.Training(1000, train_x, train_y);

    bool okay = train_x.size() == 1000 && train_y.size() == 1000;

    if (not okay) {
        std::cout << "Error: test_x.size() = " << train_x.size()
                  << ", test_y.size() = " << train_y.size() << std::endl;
    }

    for (auto i = 0; i < train_x.size(); ++i) {
        double error_ = std::cos(train_x[i]) - train_y[i];
        if (error_ > 0.0001) {
            std::cout << "Error: train_x[i] = " << train_x[i] << ", train_y[i] = " << train_y[i]
                      << ". Error: " << error_ << std::endl;
            okay = false;
        }
    }

    DC.shuffle(train_x, train_y);
    std::cout << "Shuffled training data" << std::endl;

    for (auto i = 0; i < train_x.size(); ++i) {
        double error_ = std::cos(train_x[i]) - train_y[i];
        if (error_ > 0.0001) {
            std::cout << "Error: train_x[i] = " << train_x[i] << ", train_y[i] = " << train_y[i]
                      << ". Error: " << error_ << std::endl;
            okay = false;
        }
    }

    if (okay) {
        std::cout << "Test succeeded" << std::endl;
    }
    else {
        std::cout << "Test failed" << std::endl;
    }
}
