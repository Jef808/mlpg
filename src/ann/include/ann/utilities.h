#ifndef UTILITIES_H_
#define UTILITIES_H_

struct NullFunctor {
  template <typename T>
  void operator()(T) {}; // NOLINT
};

#endif // UTILITIES_H_
