#include "hausdorff.hpp"
#include "off_loader.hpp"
#include "vertex.hpp"

int main() {
  std::string directory =
      "C:/Users/toaxo/Desktop//MSU//4_1//Hausdorff//Code//ModelSet";
  std::map<std::string, std::vector<Vertex>> models;
  load_all_off_models(directory, models);

  /*
  std::string model_name = "airplane_0005";
  if (models.find(model_name) != models.end()) {
    const std::vector<Vertex>& vertices = models[model_name];
    print_vertices(vertices);
  } else {
    std::cerr << "Model not found: " << model_name << std::endl;
  }
  */

  std::cout << "Loaded " << models.size() << " OFF models from the directory"
            << std::endl;

  std::string m1, m2;
  m1 = get_name_of_model(models, 2);
  m2 = get_name_of_model(models, 4);
  const std::vector<Vertex>& model1_vertices = models.at(m1);
  const std::vector<Vertex>& model2_vertices = models.at(m2);
  std::cout << m1 << std::endl;
  std::cout << m2 << std::endl;
  // print_vertices(models.at(m1));
  double distance =
      earlybreak_hausdorff_distance(model1_vertices, model2_vertices);
  std::cout << "Hausdorff distance: " << distance << std::endl;

  return 0;
}