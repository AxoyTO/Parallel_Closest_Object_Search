#include "helper.hpp"

std::string getOsName() {
#ifdef _WIN32
  return "Windows 32-bit";
#elif _WIN64
  return "Windows 64-bit";
#elif __APPLE__ || __MACH__
  return "Mac OSX";
#elif __linux__
  return "Linux";
#elif __FreeBSD__
  return "FreeBSD";
#elif __unix || __unix__
  return "Unix";
#else
  return "Other";
#endif
}

int set_directory(std::string& directory) {
  std::string os = getOsName();

  if (os == "Linux") {
    directory = "/mnt/c/Users/toaxo/Desktop/MSU/4_1/Hausdorff/Code/ModelSet";
  } else if (os == "_WIN32" || os == "_WIN64") {
    directory =
        "C:/Users/"
        "toaxo/Desktop/MSU/4_1/Hausdorff/Code/ModelSet";
  } else {
    return 1;
  }

  return 0;
}