#include <cfloat>
#include "hausdorff.hpp"
#include "helper.hpp"
#include "off_loader.hpp"
#include "vertex.hpp"

int main() {
  std::string directory;
  if (!set_directory(directory)) {
    std::cout << "Directory \"" << directory << "\" is set." << std::endl;
  } else {
    std::cout << "Models directory couldn't be set!" << std::endl;
    return 1;
  }

  std::map<std::string, std::vector<Vertex>> models;
  std::map<std::string, double> results;
  load_all_off_models(directory, models);

  std::cout << "Loaded " << models.size() << " OFF models from the directory"
            << std::endl;

  std::string fixed_model_name;
  fixed_model_name = get_name_of_model(models, 2);
  const std::vector<Vertex>& fixed_model_vertices = models.at(fixed_model_name);
  models.erase(fixed_model_name);
  std::cout << " — Fixed model is: " << fixed_model_name << std::endl;

  for (auto it = models.begin(); it != models.end(); it++) {
    double distance =
        earlybreak_hausdorff_distance(fixed_model_vertices, it->second);
    results[it->first] = distance;
  }

  /*
  for (auto& model : models) {
    double distance =
        earlybreak_hausdorff_distance(fixed_model_vertices, model.second);
    results[model.first] = distance;
  }
  */

  std::string closest_model;
  double min_dist = results.begin()->second;

  for (auto& res : results) {
    std::cout << "Hausdorff distance between " << fixed_model_name << " and "
              << res.first << " is: " << res.second << std::endl;
    if (res.second < min_dist) {
      min_dist = res.second;
      closest_model = res.first;
    }
  }

  std::cout << " — Closest model to " << fixed_model_name << " is "
            << closest_model << std::endl;

  return 0;
}