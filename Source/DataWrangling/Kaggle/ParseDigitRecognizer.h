#ifndef DATA_WRANGLING_KAGGLE_PARSE_DIGIT_RECOGNIZER
#define DATA_WRANGLING_KAGGLE_PARSE_DIGIT_RECOGNIZER

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace DataWrangling
{

namespace Kaggle
{

class ParseDigitRecognizer
{
  public:

    static constexpr std::size_t row_size {28};

    using StringRow = std::vector<std::string>;

    ParseDigitRecognizer() = delete;

    //--------------------------------------------------------------------------
    /// \param[in] batch_size - Number of samples in a single batch.
    //--------------------------------------------------------------------------
    ParseDigitRecognizer(const std::size_t batch_size);

    void parse(const std::string& filename);

    inline const std::vector<std::string>& get_header() const
    {
      return header_;
    }

    inline const std::vector<std::vector<int16_t>>& get_label_batches() const
    {
      return label_batches_;
    }

    inline const std::vector<std::vector<std::vector<int16_t>>>&
      get_input_batches() const
    {
      return input_batches_;
    }

  protected:

    static StringRow split(const std::string& s, char delimiter = ',');

    static int16_t string_to_int(const std::string& s);

  private:

    StringRow header_;
    std::vector<std::vector<int16_t>> label_batches_;

    // Essentially a tensor of signature (Total number of batches, N, D)
    // where N is the number of samples in a single batch and
    // D is the number of "features" (in this case 28x28=784)
    std::vector<std::vector<std::vector<int16_t>>> input_batches_;

    std::size_t batch_size_;
};

} // namespace Kaggle

} // namespace DataWrangling

#endif // DATA_WRANGLING_KAGGLE_PARSE_DIGIT_RECOGNIZER