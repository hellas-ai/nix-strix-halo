// CPU-only checks of the actual patched env.cpp, included by the test driver.
#include <cassert>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

static void reject(const std::string& value) {
  assert(setenv("MSCCLPP_IB_TC", value.c_str(), 1) == 0);
  try {
    (void)readIbTrafficClass();
    throw std::runtime_error("Malformed traffic class accepted");
  } catch (const mscclpp::Error& error) {
    assert(error.getErrorCode() == mscclpp::ErrorCode::InvalidUsage);
  }
}

int main(int argc, char** argv) {
  if (argc == 3 && std::string(argv[1]) == "constructor") {
    if (std::string(argv[2]) == "UNSET") assert(unsetenv("MSCCLPP_IB_TC") == 0);
    else assert(setenv("MSCCLPP_IB_TC", argv[2], 1) == 0);
    try {
      auto config = mscclpp::env();
      const auto first = config->ibTrafficClass;
      assert(first <= 255);
      assert(setenv("MSCCLPP_IB_TC", first == 0 ? "106" : "0", 1) == 0);
      assert(mscclpp::env()->ibTrafficClass == first);  // Read once, like other Env fields.
      std::cout << "{\"status\":\"ACCEPT\",\"value\":" << first << ",\"cached_once\":true}\n";
    } catch (const mscclpp::Error& error) {
      assert(error.getErrorCode() == mscclpp::ErrorCode::InvalidUsage);
      std::cout << "{\"status\":\"REJECT_INVALID_USAGE\"}\n";
    }
    return 0;
  }
  assert(argc == 1);
  assert(unsetenv("MSCCLPP_IB_TC") == 0);
  assert(readIbTrafficClass() == 0);
  for (unsigned value = 0; value <= 255; ++value) {
    assert(setenv("MSCCLPP_IB_TC", std::to_string(value).c_str(), 1) == 0);
    assert(readIbTrafficClass() == value);
  }
  for (const auto& pair : std::vector<std::pair<std::string, unsigned>>{
           {"00", 0}, {"000106", 106}, {"000255", 255}, {std::string(1000, '0') + "106", 106}}) {
    assert(setenv("MSCCLPP_IB_TC", pair.first.c_str(), 1) == 0);
    assert(readIbTrafficClass() == pair.second);
  }
  std::vector<std::string> invalid = {
      "", "-1", "+1", "-0", "+0", " 106", "106 ", "\t106", "106\n", "1\r", "1 06", "106junk",
      "0x6a", "0xff", "1e2", "1.0", "256", "257", "999", "000256", "65536", "4294967296",
      "18446744073709551616", std::string(1000, '9'), "106/", "\xc2\xa0" "106", "\xff", "\xd9\xa2"};
  for (const auto& value : invalid) reject(value);
  std::cout << "{\"status\":\"PASS_ACTUAL_PARSER\",\"unset\":1,\"all_byte_values\":256,"
            << "\"leading_zero_cases\":4,\"invalid_cases\":" << invalid.size() << "}\n";
}
