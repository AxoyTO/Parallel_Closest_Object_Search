#include "hausdorff.hpp"

double euclidean_distance(const Vertex& a, const Vertex& b) {
  return std::sqrt(std::pow(a.x - b.x, 2) + std::pow(a.y - b.y, 2) +
                   std::pow(a.z - b.z, 2));
}

double directed_hausdorff(const std::vector<Vertex>& model1,
                          const std::vector<Vertex>& model2) {
  double max_distance = std::numeric_limits<double>::min();

  for (const Vertex& vertex1 : model1) {
    double min_distance = std::numeric_limits<double>::max();
    for (const Vertex& vertex2 : model2) {
      double distance = euclidean_distance(vertex1, vertex2);
      min_distance = std::min(min_distance, distance);
    }
    max_distance = std::max(max_distance, min_distance);
  }

  return max_distance;
}

double hausdorff_distance(const std::vector<Vertex>& model1,
                          const std::vector<Vertex>& model2) {
  double d1 = directed_hausdorff(model1, model2);
  double d2 = directed_hausdorff(model2, model1);
  return std::max(d1, d2);
}

double earlybreak_hausdorff_distance(const std::vector<Vertex>& model1,
                                     const std::vector<Vertex>& model2) {
  double max_so_far = 0.0f;
  for (const Vertex& vertex1 : model1) {
    double min_dist1 = std::numeric_limits<double>::max();
    for (const Vertex& vertex2 : model2) {
      double dist = euclidean_distance(vertex1, vertex2);
      min_dist1 = std::min(min_dist1, dist);
      if (min_dist1 <= max_so_far) {
        break;
      }
    }
    max_so_far = std::max(max_so_far, min_dist1);
  }

  double max_dist2 = 0.0f;
  for (const Vertex& vertex2 : model2) {
    double min_dist2 = std::numeric_limits<double>::max();
    for (const Vertex& vertex1 : model1) {
      double dist = euclidean_distance(vertex1, vertex2);
      min_dist2 = std::min(min_dist2, dist);
      if (min_dist2 <= max_dist2) {
        break;
      }
    }
    max_dist2 = std::max(max_dist2, min_dist2);
  }

  return std::max(max_so_far, max_dist2);
}