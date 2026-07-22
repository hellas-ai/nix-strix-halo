#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <fcntl.h>
#include <glob.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static volatile sig_atomic_t running = 1;

static void stop_running(int signal_number)
{
    (void)signal_number;
    running = 0;
}

static uint16_t read_le16(const uint8_t *data, size_t offset)
{
    return (uint16_t)data[offset] | ((uint16_t)data[offset + 1] << 8);
}

static uint32_t read_le32(const uint8_t *data, size_t offset)
{
    return (uint32_t)data[offset] |
           ((uint32_t)data[offset + 1] << 8) |
           ((uint32_t)data[offset + 2] << 16) |
           ((uint32_t)data[offset + 3] << 24);
}

static void emit_u16_scaled(FILE *out, const char *metric, const char *label,
                            uint16_t value, double scale)
{
    if (value != UINT16_MAX)
        fprintf(out, "%s{%s} %.9g\n", metric, label, value * scale);
}

static void emit_u32_scaled(FILE *out, const char *metric, const char *label,
                            uint32_t value, double scale)
{
    if (value != UINT32_MAX)
        fprintf(out, "%s{%s} %.9g\n", metric, label, value * scale);
}

static void emit_temperature(FILE *out, const char *label, uint16_t value)
{
    /* This APU reports zero for temperature channels absent from its table. */
    if (value != 0 && value != UINT16_MAX)
        fprintf(out, "amd_smu_temperature_celsius{%s} %.9g\n",
                label, value * 0.01);
}

static int discover_metrics_path(char *path, size_t path_size)
{
    glob_t matches = {0};
    int result = glob("/sys/class/drm/card*/device/gpu_metrics", 0, NULL,
                      &matches);

    if (result != 0 || matches.gl_pathc == 0) {
        globfree(&matches);
        errno = ENOENT;
        return -1;
    }

    if (snprintf(path, path_size, "%s", matches.gl_pathv[0]) >=
        (int)path_size) {
        globfree(&matches);
        errno = ENAMETOOLONG;
        return -1;
    }

    globfree(&matches);
    return 0;
}

static int read_metrics(const char *path, uint8_t *data, size_t capacity,
                        size_t *length)
{
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    ssize_t total = 0;

    if (fd < 0)
        return -1;

    while ((size_t)total < capacity) {
        ssize_t count = read(fd, data + total, capacity - (size_t)total);
        if (count < 0) {
            if (errno == EINTR)
                continue;
            close(fd);
            return -1;
        }
        if (count == 0)
            break;
        total += count;
    }

    close(fd);
    *length = (size_t)total;
    return 0;
}

static void emit_metrics(FILE *out, const uint8_t *data, size_t length)
{
    static const char *clock_domains[] = {
        "gfx", "soc", "vpe", "ipu", "fclk", "vcn", "memory", "mpipu"
    };
    static const char *throttle_reasons[] = {
        "prochot", "spl", "fast_ppt", "slow_ppt",
        "thermal_core", "thermal_gfx", "thermal_soc"
    };
    uint16_t structure_size = read_le16(data, 0);
    unsigned int core;
    unsigned int domain;

    fprintf(out, "# HELP amd_smu_exporter_up Whether a supported AMDGPU metrics table was read.\n");
    fprintf(out, "# TYPE amd_smu_exporter_up gauge\n");
    fprintf(out, "amd_smu_exporter_up 1\n");
    fprintf(out, "# HELP amd_smu_metrics_info AMDGPU metrics table revision and size.\n");
    fprintf(out, "# TYPE amd_smu_metrics_info gauge\n");
    fprintf(out,
            "amd_smu_metrics_info{format_revision=\"%u\",content_revision=\"%u\",structure_size=\"%u\"} 1\n",
            data[2], data[3], structure_size);
    fprintf(out, "# HELP amd_smu_metrics_timestamp_seconds Unix time of the latest successful sample.\n");
    fprintf(out, "# TYPE amd_smu_metrics_timestamp_seconds gauge\n");
    fprintf(out, "amd_smu_metrics_timestamp_seconds %lld\n",
            (long long)time(NULL));

    fprintf(out, "# HELP amd_smu_temperature_celsius Temperature reported by the SMU.\n");
    fprintf(out, "# TYPE amd_smu_temperature_celsius gauge\n");
    emit_temperature(out, "domain=\"gfx\"", read_le16(data, 4));
    emit_temperature(out, "domain=\"soc\"", read_le16(data, 6));
    for (core = 0; core < 16; ++core) {
        char label[32];
        snprintf(label, sizeof(label), "domain=\"core\",core=\"%u\"", core);
        emit_temperature(out, label, read_le16(data, 8 + core * 2));
    }
    emit_temperature(out, "domain=\"skin\"", read_le16(data, 40));

    fprintf(out, "# HELP amd_smu_activity_percent Time-filtered activity reported by the SMU.\n");
    fprintf(out, "# TYPE amd_smu_activity_percent gauge\n");
    emit_u16_scaled(out, "amd_smu_activity_percent", "domain=\"gfx\"",
                    read_le16(data, 42), 1.0);
    emit_u16_scaled(out, "amd_smu_activity_percent", "domain=\"vcn\"",
                    read_le16(data, 44), 1.0);
    for (domain = 0; domain < 8; ++domain) {
        char label[32];
        snprintf(label, sizeof(label), "domain=\"ipu\",unit=\"%u\"", domain);
        emit_u16_scaled(out, "amd_smu_activity_percent", label,
                        read_le16(data, 46 + domain * 2), 1.0);
    }
    for (core = 0; core < 16; ++core) {
        char label[32];
        snprintf(label, sizeof(label), "domain=\"cpu_c0\",core=\"%u\"", core);
        emit_u16_scaled(out, "amd_smu_activity_percent", label,
                        read_le16(data, 62 + core * 2), 1.0);
    }

    fprintf(out, "# HELP amd_smu_dram_bandwidth_bytes_per_second Time-filtered DRAM traffic.\n");
    fprintf(out, "# TYPE amd_smu_dram_bandwidth_bytes_per_second gauge\n");
    emit_u16_scaled(out, "amd_smu_dram_bandwidth_bytes_per_second",
                    "direction=\"read\"", read_le16(data, 94), 1000000.0);
    emit_u16_scaled(out, "amd_smu_dram_bandwidth_bytes_per_second",
                    "direction=\"write\"", read_le16(data, 96), 1000000.0);

    fprintf(out, "# HELP amd_smu_power_watts Time-filtered power reported by the SMU.\n");
    fprintf(out, "# TYPE amd_smu_power_watts gauge\n");
    emit_u32_scaled(out, "amd_smu_power_watts", "domain=\"socket\"",
                    read_le32(data, 112), 0.001);
    emit_u16_scaled(out, "amd_smu_power_watts", "domain=\"ipu\"",
                    read_le16(data, 116), 0.001);
    emit_u32_scaled(out, "amd_smu_power_watts", "domain=\"apu\"",
                    read_le32(data, 120), 0.001);
    emit_u32_scaled(out, "amd_smu_power_watts", "domain=\"gfx\"",
                    read_le32(data, 124), 0.001);
    emit_u32_scaled(out, "amd_smu_power_watts", "domain=\"dgpu\"",
                    read_le32(data, 128), 0.001);
    emit_u32_scaled(out, "amd_smu_power_watts", "domain=\"cpu_cores\"",
                    read_le32(data, 132), 0.001);
    for (core = 0; core < 16; ++core) {
        char label[32];
        snprintf(label, sizeof(label), "domain=\"cpu_core\",core=\"%u\"", core);
        emit_u16_scaled(out, "amd_smu_power_watts", label,
                        read_le16(data, 136 + core * 2), 0.001);
    }
    emit_u16_scaled(out, "amd_smu_power_watts", "domain=\"system\"",
                    read_le16(data, 168), 0.001);

    fprintf(out, "# HELP amd_smu_clock_hertz Time-filtered target clock reported by the SMU.\n");
    fprintf(out, "# TYPE amd_smu_clock_hertz gauge\n");
    for (domain = 0; domain < 8; ++domain) {
        char label[32];
        snprintf(label, sizeof(label), "domain=\"%s\"", clock_domains[domain]);
        emit_u16_scaled(out, "amd_smu_clock_hertz", label,
                        read_le16(data, 174 + domain * 2), 1000000.0);
    }
    for (core = 0; core < 16; ++core) {
        char label[32];
        snprintf(label, sizeof(label), "domain=\"cpu_core\",core=\"%u\"", core);
        emit_u16_scaled(out, "amd_smu_clock_hertz", label,
                        read_le16(data, 190 + core * 2), 1000000.0);
    }

    fprintf(out, "# HELP amd_smu_frequency_limit_hertz Frequency limit currently enforced by firmware.\n");
    fprintf(out, "# TYPE amd_smu_frequency_limit_hertz gauge\n");
    emit_u16_scaled(out, "amd_smu_frequency_limit_hertz", "domain=\"cpu\"",
                    read_le16(data, 222), 1000000.0);
    emit_u16_scaled(out, "amd_smu_frequency_limit_hertz", "domain=\"gfx\"",
                    read_le16(data, 224), 1000000.0);

    fprintf(out, "# HELP amd_smu_throttle_residency_ticks_total Cumulative firmware PM timer ticks spent throttling.\n");
    fprintf(out, "# TYPE amd_smu_throttle_residency_ticks_total counter\n");
    for (domain = 0; domain < 7; ++domain) {
        char label[48];
        snprintf(label, sizeof(label), "reason=\"%s\"", throttle_reasons[domain]);
        emit_u32_scaled(out, "amd_smu_throttle_residency_ticks_total", label,
                        read_le32(data, 228 + domain * 4), 1.0);
    }

    (void)length;
}

static int write_snapshot(const char *metrics_path, const char *output_path)
{
    uint8_t data[4096];
    char temporary_path[4096];
    size_t length = 0;
    uint16_t structure_size;
    FILE *out;
    int fd;

    if (read_metrics(metrics_path, data, sizeof(data), &length) < 0)
        return -1;
    if (length < 260) {
        errno = EPROTO;
        return -1;
    }

    structure_size = read_le16(data, 0);
    if (data[2] != 3 || data[3] != 0 || structure_size < 260 ||
        structure_size > length) {
        errno = EPROTONOSUPPORT;
        return -1;
    }

    if (snprintf(temporary_path, sizeof(temporary_path), "%s.tmp.%ld",
                 output_path, (long)getpid()) >= (int)sizeof(temporary_path)) {
        errno = ENAMETOOLONG;
        return -1;
    }

    out = fopen(temporary_path, "w");
    if (out == NULL)
        return -1;
    emit_metrics(out, data, length);
    if (fflush(out) != 0 || (fd = fileno(out)) < 0 || fsync(fd) != 0 ||
        fclose(out) != 0) {
        int saved_errno = errno;
        unlink(temporary_path);
        errno = saved_errno;
        return -1;
    }
    if (rename(temporary_path, output_path) != 0) {
        int saved_errno = errno;
        unlink(temporary_path);
        errno = saved_errno;
        return -1;
    }

    return 0;
}

static int print_watch(const char *metrics_path)
{
    uint8_t data[4096];
    size_t length = 0;

    if (read_metrics(metrics_path, data, sizeof(data), &length) < 0)
        return -1;
    if (length < 260 || data[2] != 3 || data[3] != 0 ||
        read_le16(data, 0) < 260 || read_le16(data, 0) > length) {
        errno = EPROTONOSUPPORT;
        return -1;
    }

    if (isatty(STDOUT_FILENO))
        printf("\033[H\033[J");
    printf("AMDGPU SMU metrics v%u.%u\n", data[2], data[3]);
    printf("temperature  gfx %6.2f C   soc %6.2f C\n",
           read_le16(data, 4) / 100.0, read_le16(data, 6) / 100.0);
    printf("activity     gfx %6u %%   vcn %6u %%\n",
           read_le16(data, 42), read_le16(data, 44));
    printf("dram         read %5u MB/s   write %5u MB/s\n",
           read_le16(data, 94), read_le16(data, 96));
    printf("power        socket %6.2f W   apu %6.2f W   gfx %6.2f W   cpu %6.2f W\n",
           read_le32(data, 112) / 1000.0, read_le32(data, 120) / 1000.0,
           read_le32(data, 124) / 1000.0, read_le32(data, 132) / 1000.0);
    printf("clock        gfx %5u MHz   soc %5u MHz   fclk %5u MHz   memory %5u MHz\n",
           read_le16(data, 174), read_le16(data, 176),
           read_le16(data, 182), read_le16(data, 186));
    printf("limit        gfx %5u MHz   cpu %5u MHz\n",
           read_le16(data, 224), read_le16(data, 222));
    printf("throttle     prochot %u   spl %u   fast_ppt %u   slow_ppt %u\n",
           read_le32(data, 228), read_le32(data, 232),
           read_le32(data, 236), read_le32(data, 240));
    printf("thermal      core %u   gfx %u   soc %u\n",
           read_le32(data, 244), read_le32(data, 248),
           read_le32(data, 252));
    fflush(stdout);
    return 0;
}

static void usage(const char *program)
{
    fprintf(stderr,
            "usage: %s (--output PATH | --watch) [--gpu-metrics PATH] [--interval-ms N]\n",
            program);
}

int main(int argc, char **argv)
{
    char discovered_path[4096];
    const char *metrics_path = NULL;
    const char *output_path = NULL;
    long interval_ms = 1000;
    int watch = 0;
    struct sigaction action = {0};
    int index;

    for (index = 1; index < argc; ++index) {
        if (strcmp(argv[index], "--output") == 0 && index + 1 < argc)
            output_path = argv[++index];
        else if (strcmp(argv[index], "--watch") == 0)
            watch = 1;
        else if (strcmp(argv[index], "--gpu-metrics") == 0 && index + 1 < argc)
            metrics_path = argv[++index];
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
    if (metrics_path == NULL) {
        if (discover_metrics_path(discovered_path, sizeof(discovered_path)) < 0) {
            perror("discovering AMDGPU gpu_metrics");
            return EXIT_FAILURE;
        }
        metrics_path = discovered_path;
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

        int result = watch ? print_watch(metrics_path)
                           : write_snapshot(metrics_path, output_path);
        if (result < 0)
            fprintf(stderr, "reading %s: %s\n", metrics_path, strerror(errno));

        while (running && nanosleep(&delay, &delay) < 0 && errno == EINTR)
            ;
    }

    return EXIT_SUCCESS;
}
