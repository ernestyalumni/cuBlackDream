#ifndef DATA_WRANGLING_KAGGLE_TRANSFORM_DIGIT_RECOGNIZER_DATA
#define DATA_WRANGLING_KAGGLE_TRANSFORM_DIGIT_RECOGNIZER_DATA

#include <cstdint>
#include <utility>
#include <vector>

namespace DataWrangling
{

namespace Kaggle
{

class TransformDigitRecognizerData
{
  public:

    static std::vector<std::vector<float>> normalize_as_float(
      const std::vector<std::vector<int16_t>>& images_batch);

    //--------------------------------------------------------------------------
    /// \details embed in machine learning means to transform a single integer
    /// value, in this case the number representing a digit, e.g. 9, into an
    /// array y of size 10 such that y[i] = 1 if the digit is i and y[i] = 0 if
    /// not. But embed means something specific in differential topology so I
    /// don't like saying it. 
    //--------------------------------------------------------------------------
    static std::vector<std::vector<float>> embed_digit_as_vector(
      const std::vector<int16_t>& digits_batch);

    static std::vector<float> one_hot_encode(const int16_t digit);
};

} // namespace Kaggle
} // namespace DataWrangling

#endif // DATA_WRANGLING_KAGGLE_TRANSFORM_DIGIT_RECOGNIZER_DATA