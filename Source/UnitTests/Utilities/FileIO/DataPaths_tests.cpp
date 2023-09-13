#include "Utilities/FileIO/DataPaths.h"
#include "gtest/gtest.h"

#include <string>

using Utilities::FileIO::DataPaths;
using std::string;

namespace GoogleUnitTests
{
namespace Utilities
{
namespace FileIO
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(DataPathsTests, ProjectPathReferencesThisProject)
{
  EXPECT_TRUE(
    DataPaths::get_project_path().string().find("cuBlackDream") !=
      std::string::npos);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(DataPathsTests, DataPathHasDataAndProjectName)
{
  const auto data_path_str = DataPaths::get_data_path().string();

  EXPECT_TRUE(data_path_str.find("cuBlackDream") != string::npos);
  EXPECT_TRUE(data_path_str.find("Data") != string::npos);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(DataPathsTests, GetKaggleDataPathGetsPath)
{
  const auto kaggle_data_path_str = DataPaths::get_kaggle_data_path().string();

  EXPECT_TRUE(kaggle_data_path_str.find("cuBlackDream") != string::npos);
  EXPECT_TRUE(kaggle_data_path_str.find("Data") != string::npos);
  EXPECT_TRUE(kaggle_data_path_str.find("Kaggle") != string::npos);
}

} // namespace FileIO
} // namespace Utilities
} // namespace GoogleUnitTests