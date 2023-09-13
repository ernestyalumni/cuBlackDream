#ifndef UTILITIES_FILE_IO_DATA_PATHS
#define UTILITIES_FILE_IO_DATA_PATHS

#include <filesystem>
#include <string>

namespace Utilities
{

namespace FileIO
{

class DataPaths
{
  public:

    static constexpr char data_subdirectory_name[] {"Data"};

    static constexpr char kaggle_subdirectory_name[] {"Kaggle"};

    inline static std::filesystem::path get_project_path()
    {
      return std::filesystem::path(__FILE__).parent_path().parent_path()
        .parent_path().parent_path();
    }

    inline static std::filesystem::path get_data_path()
    {
      return get_project_path() / std::string{data_subdirectory_name};
    }

    inline static std::filesystem::path get_kaggle_data_path()
    {
      return get_project_path() / std::string{data_subdirectory_name} /
        std::string(kaggle_subdirectory_name);
    }
};

} // namespace FileIO

} // namespace Utilities

#endif // UTILITIES_FILE_IO_DATA_PATHS