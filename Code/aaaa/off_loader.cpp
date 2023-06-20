#include "off_loader.hpp"

void load_all_off_models(const std::string& directory,
                         std::map<std::string, std::vector<Vertex>>& models) {
  DIR* dir = opendir(directory.c_str());
  if (!dir) {
    std::cerr << "Error opening directory: " << directory << std::endl;
    return;
  }

  struct dirent* entry;
  int k = 0;
  while ((entry = readdir(dir)) != NULL && k != 15) {
    std::string filename = entry->d_name;
    size_t ext_pos = filename.rfind(".off");
    if (ext_pos != std::string::npos && ext_pos + 4 == filename.size()) {
      std::string model_name = filename.substr(0, ext_pos);
      std::vector<Vertex> vertices;
      load_off_model(directory + "/" + filename, vertices);
      // std::cout << "Loaded " << model_name << " with " << vertices.size()
      //           << " vertices from the OFF file" << std::endl;
      models[model_name] = vertices;
      k++;
    }
  }

  closedir(dir);
}

void load_off_model(const std::string& filename,
                    std::vector<Vertex>& vertices) {
  std::ifstream file(filename);

  if (!file.is_open()) {
    std::cerr << "Could not open the file: " << filename << std::endl;
    return;
  }

  std::string line;
  std::getline(file, line);  // Read the first line (OFF header)

  if (line != "OFF") {
    std::cerr << "The file is not in OFF format" << std::endl;
    return;
  }

  int num_vertices, num_faces, num_edges;
  file >> num_vertices >> num_faces >> num_edges;  // Read counts

  vertices.reserve(num_vertices);

  for (int i = 0; i < num_vertices; ++i) {
    Vertex vertex;
    file >> vertex.x >> vertex.y >> vertex.z;
    vertices.push_back(vertex);
  }
}

void print_vertices(const std::vector<Vertex>& vertices) {
  for (const auto& vertex : vertices) {
    std::cout << vertex.x << " " << vertex.y << " " << vertex.z << std::endl;
  }
}

std::string get_name_of_model(
    std::map<std::string, std::vector<Vertex>>& models,
    const int& index) {
  auto it = models.begin();
  std::advance(it, index);
  return it->first;
}