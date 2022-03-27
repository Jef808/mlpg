#include <iostream>
#include <LinearAnn.h>
#include "Config.h"

namespace simple {


void config_header(Config config) {
    if constexpr (std::is_signed_v<Eigen::Index>)
      std::cout << "Warning! The type deduced for Config::Index is signed!" << std::endl;
}

void show_compilation_timestamp(std::ostream& out) {
  out << "@@@\n  " << __DATE__ << " | " __TIME__ << "\n@@@";
}

}  // simple
