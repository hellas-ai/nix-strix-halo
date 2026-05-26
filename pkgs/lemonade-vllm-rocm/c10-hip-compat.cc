#include <cstdint>

struct C10Stream {
  uint64_t opaque;
};

struct C10SourceLocation {
  const char *function;
  const char *file;
  uint32_t line;
};

extern "C" int8_t cuda_current_device()
    __asm__("_ZN3c104cuda14current_deviceEv");
extern "C" C10Stream cuda_get_stream_from_pool_bool(bool, int8_t)
    __asm__("_ZN3c104cuda17getStreamFromPoolEba");
extern "C" C10Stream cuda_get_stream_from_pool_int(int, int8_t)
    __asm__("_ZN3c104cuda17getStreamFromPoolEia");
extern "C" C10Stream cuda_get_current_stream(int8_t)
    __asm__("_ZN3c104cuda20getCurrentCUDAStreamEa");
extern "C" C10Stream cuda_get_default_stream(int8_t)
    __asm__("_ZN3c104cuda20getDefaultCUDAStreamEa");
extern "C" void cuda_set_current_stream(C10Stream)
    __asm__("_ZN3c104cuda20setCurrentCUDAStreamENS0_10CUDAStreamE");
extern "C" void cuda_warn_or_error_on_sync()
    __asm__("_ZN3c104cuda21warn_or_error_on_syncEv");
extern "C" void cuda_check_impl(int, const char *, const char *, uint32_t, bool)
    __asm__("_ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_jb");
extern "C" void *cuda_stream_stream(const C10Stream *)
    __asm__("_ZNK3c104cuda10CUDAStream6streamEv");
extern "C" void c10_message_logger_new_ctor(
    void *, C10SourceLocation, int, bool)
    __asm__("_ZN3c1013MessageLoggerC1ENS_14SourceLocationEib");

extern "C" int8_t hip_current_device()
    __asm__("_ZN3c103hip14current_deviceEv");
extern "C" C10Stream hip_get_stream_from_pool_bool(bool, int8_t)
    __asm__("_ZN3c103hip17getStreamFromPoolEba");
extern "C" C10Stream hip_get_stream_from_pool_int(int, int8_t)
    __asm__("_ZN3c103hip17getStreamFromPoolEia");
extern "C" C10Stream hip_get_current_stream(int8_t)
    __asm__("_ZN3c103hip19getCurrentHIPStreamEa");
extern "C" C10Stream hip_get_default_stream(int8_t)
    __asm__("_ZN3c103hip19getDefaultHIPStreamEa");
extern "C" void hip_set_current_stream(C10Stream)
    __asm__("_ZN3c103hip19setCurrentHIPStreamENS0_9HIPStreamE");
extern "C" void hip_warn_or_error_on_sync()
    __asm__("_ZN3c103hip21warn_or_error_on_syncEv");
extern "C" void hip_check_impl(int, const char *, const char *, uint32_t, bool)
    __asm__("_ZN3c103hip28c10_hip_check_implementationEiPKcS2_jb");
extern "C" void *hip_stream_stream(const C10Stream *)
    __asm__("_ZNK3c103hip9HIPStream6streamEv");
extern "C" void c10_message_logger_old_ctor(void *, const char *, int, int, bool)
    __asm__("_ZN3c1013MessageLoggerC1EPKciib");

extern "C" int8_t hip_current_device() { return cuda_current_device(); }

extern "C" C10Stream hip_get_stream_from_pool_bool(
    bool is_high_priority, int8_t device) {
  return cuda_get_stream_from_pool_bool(is_high_priority, device);
}

extern "C" C10Stream hip_get_stream_from_pool_int(
    int priority, int8_t device) {
  return cuda_get_stream_from_pool_int(priority, device);
}

extern "C" C10Stream hip_get_current_stream(int8_t device) {
  return cuda_get_current_stream(device);
}

extern "C" C10Stream hip_get_default_stream(int8_t device) {
  return cuda_get_default_stream(device);
}

extern "C" void hip_set_current_stream(C10Stream stream) {
  cuda_set_current_stream(stream);
}

extern "C" void hip_warn_or_error_on_sync() { cuda_warn_or_error_on_sync(); }

extern "C" void hip_check_impl(
    int code,
    const char *file,
    const char *func,
    uint32_t line,
    bool include_device_assertions) {
  cuda_check_impl(code, file, func, line, include_device_assertions);
}

extern "C" void *hip_stream_stream(const C10Stream *stream) {
  return cuda_stream_stream(stream);
}

extern "C" void c10_message_logger_old_ctor(
    void *self, const char *file, int line, int severity, bool exit_on_fatal) {
  C10SourceLocation loc{nullptr, file, static_cast<uint32_t>(line)};
  c10_message_logger_new_ctor(self, loc, severity, exit_on_fatal);
}
