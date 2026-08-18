/*
 * amd-npu-exporter: Prometheus textfile exporter for the AMD XDNA NPU
 * (Ryzen AI, "IPU"/AIE) as exposed by the in-kernel amdxdna driver.
 *
 * The driver's only telemetry interface is the DRM_IOCTL_AMDXDNA_GET_INFO
 * ioctl on the accel render node. There is no sysfs or debugfs telemetry, and
 * DRM_AMDXDNA_QUERY_SENSORS returns -ENODEV on Strix Halo, so this exporter
 * deliberately publishes no NPU power or temperature -- the hardware does not
 * report them. Every metric below was verified to return real data on a
 * Strix Halo (AMD Ryzen AI MAX+ 395) running kernel 7.x / amdxdna 0.10.0.
 *
 * The queries used are grouped by the kernel uapi that first exposed them:
 *   - always (amdxdna since v6.14): AIE_METADATA, AIE_VERSION,
 *     CLOCK_METADATA, FIRMWARE_VERSION.
 *   - v7.0+ (compiled only when the fed-in uapi header has them, via the
 *     HAVE_NPU_* macros the derivation sets): GET_POWER_MODE, RESOURCE_INFO,
 *     and hardware-context occupancy via GET_ARRAY.
 *
 * Output mirrors amdgpu-smu-exporter: an atomically-replaced .prom file for
 * node_exporter's textfile collector, or a --watch pretty-printer.
 */
#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

/* The amdxdna uapi drags in <linux/types.h> (__int128) and <linux/stddef.h>
 * (named variadic macros), which are not ISO C; scope the -Wpedantic noise to
 * the kernel headers without relaxing it for our own code. */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wvariadic-macros"
#include <drm/amdxdna_accel.h>
#pragma GCC diagnostic pop

static volatile sig_atomic_t running = 1;

static void stop_running(int signal_number)
{
    (void)signal_number;
    running = 0;
}

/* Issue one GET_INFO query. Returns 0 on success; on failure the caller can
 * inspect errno (a query the firmware does not implement returns -ENODEV or
 * -EINVAL, which we treat as "metric absent", not as exporter failure). */
static int get_info(int fd, uint32_t param, void *buffer, uint32_t size)
{
    struct amdxdna_drm_get_info info = {
        .param = param,
        .buffer_size = size,
        .buffer = (uintptr_t)buffer,
    };
    return ioctl(fd, DRM_IOCTL_AMDXDNA_GET_INFO, &info);
}

/* --- metric emission ----------------------------------------------------- */

struct npu {
    /* static identity, read once on first successful open */
    int identified;
    char firmware_version[32];
    char aie_version[16];
    uint16_t columns;
    uint16_t rows;
#ifdef HAVE_NPU_RESOURCE_INFO
    uint64_t clk_max_mhz;
    uint64_t tops_max;
    uint64_t tasks_max;
#endif
};

static void read_identity(int fd, struct npu *s)
{
    struct amdxdna_drm_query_firmware_version fw = {0};
    struct amdxdna_drm_query_aie_version aie = {0};
    struct amdxdna_drm_query_aie_metadata md = {0};

    if (get_info(fd, DRM_AMDXDNA_QUERY_FIRMWARE_VERSION, &fw, sizeof fw) == 0)
        snprintf(s->firmware_version, sizeof s->firmware_version,
                 "%u.%u.%u.%u", fw.major, fw.minor, fw.patch, fw.build);
    else
        snprintf(s->firmware_version, sizeof s->firmware_version, "unknown");

    if (get_info(fd, DRM_AMDXDNA_QUERY_AIE_VERSION, &aie, sizeof aie) == 0)
        snprintf(s->aie_version, sizeof s->aie_version, "%u.%u", aie.major,
                 aie.minor);
    else
        snprintf(s->aie_version, sizeof s->aie_version, "unknown");

    if (get_info(fd, DRM_AMDXDNA_QUERY_AIE_METADATA, &md, sizeof md) == 0) {
        s->columns = md.cols;
        s->rows = md.rows;
    }

#ifdef HAVE_NPU_RESOURCE_INFO
    {
        struct amdxdna_drm_get_resource_info ri = {0};
        if (get_info(fd, DRM_AMDXDNA_QUERY_RESOURCE_INFO, &ri, sizeof ri) == 0) {
            s->clk_max_mhz = ri.npu_clk_max;
            s->tops_max = ri.npu_tops_max;
            s->tasks_max = ri.npu_task_max;
        }
    }
#endif
    s->identified = 1;
}

static void emit_identity(FILE *out, const struct npu *s)
{
    fprintf(out, "# HELP amd_npu_info AMD XDNA NPU identity. Always 1.\n");
    fprintf(out, "# TYPE amd_npu_info gauge\n");
    fprintf(out, "amd_npu_info{firmware_version=\"%s\",aie_version=\"%s\"} 1\n",
            s->firmware_version, s->aie_version);

    if (s->columns) {
        fprintf(out, "# HELP amd_npu_columns Total AIE array columns.\n");
        fprintf(out, "# TYPE amd_npu_columns gauge\n");
        fprintf(out, "amd_npu_columns %u\n", s->columns);
        fprintf(out, "# HELP amd_npu_rows Total AIE array rows.\n");
        fprintf(out, "# TYPE amd_npu_rows gauge\n");
        fprintf(out, "amd_npu_rows %u\n", s->rows);
    }

#ifdef HAVE_NPU_RESOURCE_INFO
    if (s->clk_max_mhz) {
        fprintf(out, "# HELP amd_npu_clock_max_hertz Maximum H-clock frequency.\n");
        fprintf(out, "# TYPE amd_npu_clock_max_hertz gauge\n");
        fprintf(out, "amd_npu_clock_max_hertz %.9g\n",
                (double)s->clk_max_mhz * 1e6);
        fprintf(out, "# HELP amd_npu_tops_max Maximum NPU TOPs.\n");
        fprintf(out, "# TYPE amd_npu_tops_max gauge\n");
        fprintf(out, "amd_npu_tops_max %llu\n", (unsigned long long)s->tops_max);
        fprintf(out, "# HELP amd_npu_tasks_max Maximum concurrent NPU tasks.\n");
        fprintf(out, "# TYPE amd_npu_tasks_max gauge\n");
        fprintf(out, "amd_npu_tasks_max %llu\n",
                (unsigned long long)s->tasks_max);
    }
#endif
}

static void emit_clocks(FILE *out, int fd)
{
    struct amdxdna_drm_query_clock_metadata ck = {0};
    if (get_info(fd, DRM_AMDXDNA_QUERY_CLOCK_METADATA, &ck, sizeof ck) != 0)
        return;
    fprintf(out, "# HELP amd_npu_clock_hertz Current NPU clock frequency.\n");
    fprintf(out, "# TYPE amd_npu_clock_hertz gauge\n");
    fprintf(out, "amd_npu_clock_hertz{domain=\"mp_npu\"} %.9g\n",
            (double)ck.mp_npu_clock.freq_mhz * 1e6);
    fprintf(out, "amd_npu_clock_hertz{domain=\"h\"} %.9g\n",
            (double)ck.h_clock.freq_mhz * 1e6);
}

#ifdef HAVE_NPU_POWER_MODE
static void emit_power_mode(FILE *out, int fd)
{
    struct amdxdna_drm_get_power_mode pm = {0};
    if (get_info(fd, DRM_AMDXDNA_GET_POWER_MODE, &pm, sizeof pm) != 0)
        return;
    fprintf(out, "# HELP amd_npu_power_mode Power mode: 0 default, 1 low, 2 medium, 3 high, 4 turbo.\n");
    fprintf(out, "# TYPE amd_npu_power_mode gauge\n");
    fprintf(out, "amd_npu_power_mode %u\n", pm.power_mode);
}
#endif

#ifdef HAVE_NPU_RESOURCE_INFO
static void emit_resource_load(FILE *out, int fd)
{
    struct amdxdna_drm_get_resource_info ri = {0};
    if (get_info(fd, DRM_AMDXDNA_QUERY_RESOURCE_INFO, &ri, sizeof ri) != 0)
        return;
    fprintf(out, "# HELP amd_npu_tops_current Currently available NPU TOPs.\n");
    fprintf(out, "# TYPE amd_npu_tops_current gauge\n");
    fprintf(out, "amd_npu_tops_current %llu\n",
            (unsigned long long)ri.npu_tops_curr);
    fprintf(out, "# HELP amd_npu_tasks_current NPU tasks currently scheduled.\n");
    fprintf(out, "# TYPE amd_npu_tasks_current gauge\n");
    fprintf(out, "amd_npu_tasks_current %llu\n",
            (unsigned long long)ri.npu_task_curr);
}
#endif

#ifdef HAVE_NPU_GET_ARRAY
/* Hardware-context occupancy via GET_ARRAY(HW_CONTEXT_ALL): num_element is a
 * true in/out count, so idle reliably reports zero contexts. We publish only
 * bounded, aggregate series -- never per-pid labels. */
static void emit_hw_contexts(FILE *out, int fd)
{
    struct amdxdna_drm_hwctx_entry entries[64];
    struct amdxdna_drm_get_array arr = {
        .param = DRM_AMDXDNA_HW_CONTEXT_ALL,
        .element_size = sizeof entries[0],
        .num_element = sizeof entries / sizeof entries[0],
        .buffer = (uintptr_t)entries,
    };
    if (ioctl(fd, DRM_IOCTL_AMDXDNA_GET_ARRAY, &arr) != 0)
        return;

    uint32_t stride = arr.element_size ? arr.element_size : sizeof entries[0];
    uint32_t count = arr.num_element;
    if (count > sizeof entries / sizeof entries[0])
        count = sizeof entries / sizeof entries[0];

    uint32_t active = 0, columns_in_use = 0;
    uint64_t submissions = 0, completions = 0, errors = 0;
    for (uint32_t i = 0; i < count; ++i) {
        const struct amdxdna_drm_hwctx_entry *e =
            (const void *)((const uint8_t *)entries + (size_t)i * stride);
        active += 1;
        columns_in_use += e->num_col;
        submissions += e->command_submissions;
        completions += e->command_completions;
        errors += e->errors;
    }

    fprintf(out, "# HELP amd_npu_hw_contexts Active NPU hardware contexts.\n");
    fprintf(out, "# TYPE amd_npu_hw_contexts gauge\n");
    fprintf(out, "amd_npu_hw_contexts %u\n", active);
    fprintf(out, "# HELP amd_npu_columns_in_use AIE columns bound to active contexts.\n");
    fprintf(out, "# TYPE amd_npu_columns_in_use gauge\n");
    fprintf(out, "amd_npu_columns_in_use %u\n", columns_in_use);
    fprintf(out, "# HELP amd_npu_command_submissions_total Commands submitted across contexts.\n");
    fprintf(out, "# TYPE amd_npu_command_submissions_total counter\n");
    fprintf(out, "amd_npu_command_submissions_total %llu\n",
            (unsigned long long)submissions);
    fprintf(out, "# HELP amd_npu_command_completions_total Commands completed across contexts.\n");
    fprintf(out, "# TYPE amd_npu_command_completions_total counter\n");
    fprintf(out, "amd_npu_command_completions_total %llu\n",
            (unsigned long long)completions);
    fprintf(out, "# HELP amd_npu_context_errors_total Context errors across contexts.\n");
    fprintf(out, "# TYPE amd_npu_context_errors_total counter\n");
    fprintf(out, "amd_npu_context_errors_total %llu\n",
            (unsigned long long)errors);
}
#endif

static void emit_all(FILE *out, int fd, struct npu *s)
{
    fprintf(out, "# HELP amd_npu_up Whether the amdxdna NPU is present and responding.\n");
    fprintf(out, "# TYPE amd_npu_up gauge\n");
    fprintf(out, "amd_npu_up 1\n");
    fprintf(out, "# HELP amd_npu_scrape_timestamp_seconds Unix time of the latest successful sample.\n");
    fprintf(out, "# TYPE amd_npu_scrape_timestamp_seconds gauge\n");
    fprintf(out, "amd_npu_scrape_timestamp_seconds %lld\n", (long long)time(NULL));

    if (!s->identified)
        read_identity(fd, s);
    emit_identity(out, s);
    emit_clocks(out, fd);
#ifdef HAVE_NPU_POWER_MODE
    emit_power_mode(out, fd);
#endif
#ifdef HAVE_NPU_RESOURCE_INFO
    emit_resource_load(out, fd);
#endif
#ifdef HAVE_NPU_GET_ARRAY
    emit_hw_contexts(out, fd);
#endif
}

/* When the NPU never probed (e.g. amdxdna SMU init failed, as seen on some
 * Strix Halo units), the render node is absent. We still publish amd_npu_up 0
 * so the down state is visible rather than a silent gap. */
static void emit_down(FILE *out)
{
    fprintf(out, "# HELP amd_npu_up Whether the amdxdna NPU is present and responding.\n");
    fprintf(out, "# TYPE amd_npu_up gauge\n");
    fprintf(out, "amd_npu_up 0\n");
}

static int write_snapshot(const char *device_path, const char *output_path,
                          struct npu *s)
{
    char temporary_path[4096];
    FILE *out;
    int fd_dev;
    int fd_out;

    if (snprintf(temporary_path, sizeof temporary_path, "%s.tmp.%ld",
                 output_path, (long)getpid()) >= (int)sizeof temporary_path) {
        errno = ENAMETOOLONG;
        return -1;
    }

    out = fopen(temporary_path, "w");
    if (out == NULL)
        return -1;

    fd_dev = open(device_path, O_RDWR | O_CLOEXEC);
    if (fd_dev < 0) {
        emit_down(out);
    } else {
        emit_all(out, fd_dev, s);
        close(fd_dev);
    }

    if (fflush(out) != 0 || (fd_out = fileno(out)) < 0 || fsync(fd_out) != 0 ||
        fclose(out) != 0) {
        int saved = errno;
        unlink(temporary_path);
        errno = saved;
        return -1;
    }
    if (rename(temporary_path, output_path) != 0) {
        int saved = errno;
        unlink(temporary_path);
        errno = saved;
        return -1;
    }
    return 0;
}

static int print_watch(const char *device_path, struct npu *s)
{
    int fd_dev = open(device_path, O_RDWR | O_CLOEXEC);
    if (isatty(STDOUT_FILENO))
        printf("\033[H\033[J");
    if (fd_dev < 0) {
        printf("amdxdna NPU: DOWN (%s: %s)\n", device_path, strerror(errno));
        fflush(stdout);
        return 0;
    }
    if (!s->identified)
        read_identity(fd_dev, s);
    struct amdxdna_drm_query_clock_metadata ck = {0};
    get_info(fd_dev, DRM_AMDXDNA_QUERY_CLOCK_METADATA, &ck, sizeof ck);
    printf("amdxdna NPU  fw %s  aie %s  cols %u\n", s->firmware_version,
           s->aie_version, s->columns);
    printf("clock        mp-npu %u MHz   h %u MHz\n", ck.mp_npu_clock.freq_mhz,
           ck.h_clock.freq_mhz);
    close(fd_dev);
    fflush(stdout);
    return 0;
}

static void usage(const char *program)
{
    fprintf(stderr,
            "usage: %s (--output PATH | --watch) [--device PATH] [--interval-ms N]\n",
            program);
}

int main(int argc, char **argv)
{
    const char *device_path = "/dev/accel/accel0";
    const char *output_path = NULL;
    long interval_ms = 1000;
    int watch = 0;
    struct sigaction action = {0};
    struct npu state = {0};
    int index;

    for (index = 1; index < argc; ++index) {
        if (strcmp(argv[index], "--output") == 0 && index + 1 < argc)
            output_path = argv[++index];
        else if (strcmp(argv[index], "--watch") == 0)
            watch = 1;
        else if (strcmp(argv[index], "--device") == 0 && index + 1 < argc)
            device_path = argv[++index];
        else if (strcmp(argv[index], "--interval-ms") == 0 && index + 1 < argc)
            interval_ms = strtol(argv[++index], NULL, 10);
        else {
            usage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    if ((output_path == NULL) == !watch || interval_ms < 100) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    action.sa_handler = stop_running;
    sigemptyset(&action.sa_mask);
    sigaction(SIGINT, &action, NULL);
    sigaction(SIGTERM, &action, NULL);

    while (running) {
        struct timespec delay = {
            .tv_sec = interval_ms / 1000,
            .tv_nsec = (interval_ms % 1000) * 1000000L,
        };

        int result = watch ? print_watch(device_path, &state)
                           : write_snapshot(device_path, output_path, &state);
        if (result < 0)
            fprintf(stderr, "%s: %s\n", output_path ? output_path : device_path,
                    strerror(errno));

        while (running && nanosleep(&delay, &delay) < 0 && errno == EINTR)
            ;
    }

    return EXIT_SUCCESS;
}
