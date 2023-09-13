#ifndef UTILITIES_TIME_WRANGLE_DATES
#define UTILITIES_TIME_WRANGLE_DATES

#include <chrono>
#include <cstdint>
#include <string>

namespace Utilities
{

namespace Time
{

class WrangleDates
{
  public:

    static std::chrono::seconds date_to_chrono_seconds(const std::string& date);

    static int64_t date_to_seconds(const std::string& date);
};

} // namespace Time

} // namespace Utilities

#endif // UTILITIES_TIME_WRANGLE_DATES