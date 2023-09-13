#include "ParseDigitRecognizer.h"

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <ranges>
#include <sstream>
#include <string>
#include <vector>

using std::size_t;
using std::string;
using std::vector;

namespace DataWrangling
{
namespace Kaggle
{

ParseDigitRecognizer::ParseDigitRecognizer(const size_t batch_size):
  header_{},
  label_batches_{},
  batch_size_{batch_size}
{}

void ParseDigitRecognizer::parse(const string& filename)
{
  // Open file for reading.
  std::ifstream file {filename};

  if (!file.is_open())
  {
    throw std::runtime_error("Failed to open file: " + filename);
  }

  // Parse the header.
  // Assume we have a header first and only for the first row.
  string line {};
  if (std::getline(file, line))
  {
    header_ = split(line);
  }

  size_t batch_counter {0};
  vector<int16_t> labels {};
  vector<vector<int16_t>> images {};

  // Parse the data.
  while (std::getline(file, line))
  {
    const StringRow parsed_line {split(line)};

    if (batch_counter >= batch_size_)
    {
      label_batches_.push_back(labels);
      labels.clear();

      input_batches_.push_back(images);
      images.clear();
    }

    labels.push_back(string_to_int(parsed_line.at(0)));

    auto pixels_data = parsed_line
      | std::views::drop(1)
      | std::views::take(parsed_line.size() - 1)
      | std::views::transform(string_to_int);
    vector<int16_t> image (pixels_data.begin(), pixels_data.end());
    images.push_back(image);

    if (batch_counter < batch_size_)
    {
      ++batch_counter;
    }
    else
    {
      batch_counter = 1;
    }
  }
}

ParseDigitRecognizer::StringRow ParseDigitRecognizer::split(
  const string& s,
  char delimiter)
{
  StringRow tokens {};
  std::istringstream token_stream {s};

  string token {};

  while (std::getline(token_stream, token, delimiter))
  {
    tokens.push_back(token);
  }

  return tokens;
}

int16_t ParseDigitRecognizer::string_to_int(const string& s)
{
  int16_t value {-1};

  auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), value);

  if (ec == std::errc())
  {
    return value;
  }
  else
  {
    std::cerr << "Failed to convert string to int16_t: " << s << std::endl;
    return -1;
  }
}

} // namespace Kaggle
} // namespace DataWrangling
