#include "DataWrangling/Kaggle/TransformDigitRecognizerData.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cstdint>
#include <numeric> // std::iota
#include <vector>

using TDRD = DataWrangling::Kaggle::TransformDigitRecognizerData;
using std::string;
using std::vector;

namespace GoogleUnitTests
{
namespace DataWrangling
{
namespace Kaggle
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TransformDigitRecognizerDataTests, NormalizeAsFloatNormalizes)
{
  vector<vector<int16_t>> pixels_batch {};

  for (int16_t i {0}; i < 16; ++i)
  {
    vector<int16_t> pixels (16);

    int16_t value {static_cast<int16_t>(i * 16)};

    std::generate_n(pixels.begin(), 16, [&value]() { return value++; });

    pixels_batch.emplace_back(pixels);
  }

  const auto result = TDRD::normalize_as_float(pixels_batch);
  EXPECT_EQ(result.size(), 16);

  for (int16_t i {0}; i < 16; ++i)
  {
    for (int16_t j {0}; j < 16; ++j)
    {
      EXPECT_FLOAT_EQ(
        result.at(i).at(j),
        static_cast<float>(i * static_cast<int16_t>(16) + j) / 256.f);
    }
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TransformDigitRecognizerDataTests, EmbedDigitAsVectorTransforms)
{
  vector<int16_t> digits (10);
  // https://en.cppreference.com/w/cpp/algorithm/iota
  // Fills range with sequentially increasing values, starting with value and
  // repetitively evaluating ++value.
  std::iota(digits.begin(), digits.end(), 0);

  const auto embedded = TDRD::embed_digit_as_vector(digits);

  for (int i {0}; i < 10; ++i)
  {
    for (int j {0}; j < 10; ++j)
    {
      if (i == j)
      {
        EXPECT_FLOAT_EQ(embedded.at(i).at(j), 1.0f);
      }
      else
      {
        EXPECT_FLOAT_EQ(embedded.at(i).at(j), 0.0f);        
      }
    }
  }
}

} // namespace Kaggle
} // namespace DataWrangling
} // namespace GoogleUnitTests