#include <iostream>

#include "include/simulation.hpp"
#include "include/vector.hpp"

using namespace kocs;

int main() {
  Vector3<float> v1{ 0.0f, 0.0f, 0.0f };
  Vector3<float> v2{ 28.0f, 0.0f, -7.0f };
  v1 -= v2;

  std::cout << v1.x() << " " << v1.y() << " " << v1.z() << std::endl;

  return 0;
}
