#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>
#include "vertex.hpp"

double euclidean_distance(const Vertex& a, const Vertex& b);

double directed_hausdorff(const std::vector<Vertex>& model1,
                          const std::vector<Vertex>& model2);

double hausdorff_distance(const std::vector<Vertex>& model1,
                          const std::vector<Vertex>& model2);

double earlybreak_hausdorff_distance(const std::vector<Vertex>& model1,
                                     const std::vector<Vertex>& model2);