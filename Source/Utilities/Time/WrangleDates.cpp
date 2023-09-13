#include "WrangleDates.h"

#include <cstdint>
#include <iostream>
#include <string>

using std::string;

namespace Utilities
{
namespace Time
{

std::chrono::seconds WrangleDates::date_to_chrono_seconds(const string& date)
{
  unsigned int year {0};
  unsigned int month {0};
  unsigned int day {0};

  std::sscanf(date.c_str(), "%d-%d-%d", &year, &month, &day);

  std::chrono::year_month_day year_month_day {
    std::chrono::year{year},
    std::chrono::month{month},
    std::chrono::day{day}};

  auto sys_days = std::chrono::sys_days(year_month_day);

  return std::chrono::duration_cast<std::chrono::seconds>(
    sys_days.time_since_epoch());
}

int64_t WrangleDates::date_to_seconds(const string& date)
{
  auto chrono_seconds = WrangleDates::date_to_chrono_seconds(date);

  return static_cast<int64_t>(chrono_seconds.count());
}

} // namespace Time
} // namespace Utilities
