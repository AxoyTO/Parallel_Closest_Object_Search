#pragma once

#include <dirent.h>
#include <omp.h>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include "vertex.hpp"

void load_all_off_models(const std::string& directory,
                         std::map<std::string, std::vector<Vertex>>& models);

void load_off_model(const std::string& filename, std::vector<Vertex>& vertices);

void print_vertices(const std::vector<Vertex>& vertices);

std::string get_name_of_model(
    std::map<std::string, std::vector<Vertex>>& models,
    const int& index);