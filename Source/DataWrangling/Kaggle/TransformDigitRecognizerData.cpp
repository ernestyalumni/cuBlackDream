#include "TransformDigitRecognizerData.h"

#include <cstdint>
#include <ranges>
#include <stdexcept>
#include <vector>

using std::vector;

namespace DataWrangling
{
namespace Kaggle
{

vector<vector<float>> TransformDigitRecognizerData::normalize_as_float(
  const vector<vector<int16_t>>& images_batch)
{
  auto transform_and_normalize = [](const vector<int16_t>& image)
  {
    return image | std::views::transform(
      [](const int value)
      {
        return static_cast<float>(value) / 256.0f;
      }); 
  };

  auto transformed_images_batch = images_batch
    | std::views::transform(transform_and_normalize);

  vector<vector<float>> result {};
  for (const auto& image : transformed_images_batch)
  {
    result.emplace_back(image.begin(), image.end());
  }

  return result;
}

vector<vector<float>> TransformDigitRecognizerData::embed_digit_as_vector(
  const vector<int16_t>& digits_batch)
{
  auto transformed_digits = digits_batch
    | std::views::transform(one_hot_encode);

  return {transformed_digits.begin(), transformed_digits.end()};
}

vector<float> TransformDigitRecognizerData::one_hot_encode(const int16_t digit)
{
  if (digit < 0 || digit > 9)
  {
    throw std::out_of_range("Digit must be between 0 and 9, inclusive");
  }

  // Initialize a vector of size 10 with all zeros.
  vector<float> encoding (10, 0.0f);

  // Set the corresponding index to 1.
  encoding[digit] = 1.0f;

  return encoding;
}

} // namespace Kaggle
} // namespace DataWrangling
