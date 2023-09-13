#include "DataWrangling/Kaggle/ParseDigitRecognizer.h"
#include "Utilities/FileIO/DataPaths.h"
#include "gtest/gtest.h"

#include <string>

using DataWrangling::Kaggle::ParseDigitRecognizer;
using Utilities::FileIO::DataPaths;
using std::string;

namespace GoogleUnitTests
{
namespace DataWrangling
{
namespace Kaggle
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(ParseDigitRecognizerTests, Constructs)
{
  ParseDigitRecognizer parser {100};

  const auto& header = parser.get_header();

  EXPECT_EQ(header.size(), 0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(ParseDigitRecognizerTests, Parses)
{
  const auto training_path = DataPaths::get_kaggle_data_path() /
    "DigitRecognizer" /
    "digit-recognizer" /
    "train.csv";

  ParseDigitRecognizer parser {100};

  parser.parse(training_path.string());

  const auto& header = parser.get_header();

  EXPECT_EQ(header.size(), 785);

  EXPECT_EQ(header.at(0), "label");
  EXPECT_EQ(header.at(1), "pixel0");
  EXPECT_EQ(header.at(2), "pixel1");
  EXPECT_EQ(header.at(3), "pixel2");
  EXPECT_EQ(header.at(784), "pixel783\r");

  const auto& labels = parser.get_label_batches();

  EXPECT_EQ(labels.size(), 419);

  for (const auto& label_batch : labels)
  {
    EXPECT_EQ(label_batch.size(), 100);
  }

  EXPECT_EQ(labels.at(0).at(0), 1);
  EXPECT_EQ(labels.at(0).at(1), 0);
  EXPECT_EQ(labels.at(0).at(2), 1);
  EXPECT_EQ(labels.at(0).at(3), 4);
  EXPECT_EQ(labels.at(0).at(4), 0);
  EXPECT_EQ(labels.at(0).at(5), 0);
  EXPECT_EQ(labels.at(0).at(6), 7);
  EXPECT_EQ(labels.at(0).at(7), 3);
  EXPECT_EQ(labels.at(0).at(8), 5);
  EXPECT_EQ(labels.at(0).at(9), 3);
  EXPECT_EQ(labels.at(0).at(10), 8);
  EXPECT_EQ(labels.at(0).at(11), 9);
  EXPECT_EQ(labels.at(0).at(12), 1);

  const auto& images = parser.get_input_batches();

  EXPECT_EQ(images.size(), 419);

  for (const auto& images_batch : images)
  {
    EXPECT_EQ(images_batch.size(), 100);

    for (const auto& image : images_batch)
    {
      EXPECT_EQ(image.size(), 784);
    }
  }

  // Batch 0, Sample 1, pixel X.
  EXPECT_EQ(images.at(0).at(1).at(148), 13);
  EXPECT_EQ(images.at(0).at(1).at(149), 86);
  EXPECT_EQ(images.at(0).at(1).at(150), 250);

  EXPECT_EQ(images.at(0).at(2).at(152), 9);
  EXPECT_EQ(images.at(0).at(2).at(153), 254);
  EXPECT_EQ(images.at(0).at(2).at(154), 254);

  EXPECT_EQ(images.at(1).at(1).at(149), 253);
  EXPECT_EQ(images.at(1).at(1).at(150), 224);
  EXPECT_EQ(images.at(1).at(1).at(151), 223);
}

} // namespace Kaggle
} // namespace DataWrangling
} // namespace GoogleUnitTests