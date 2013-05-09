#ifndef USEFUL_FUNCTIONS_H
#define USEFUL_FUNCTIONS_H

#include <string>
#include <sstream>

template <typename T>
std::string number_to_string ( T Number ){
  std::ostringstream ss;
  ss << Number;
  return ss.str();
}

template <typename T>
T string_to_number ( const std::string & Text ){
  std::istringstream ss(Text);
  T result;
  return ss >> result ? result : 0;
}

#endif
