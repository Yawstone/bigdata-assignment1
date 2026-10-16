/* Basic dense matrix multiplication, triple loop (i-j-k), C = A * B.
 * Storage: flat row-major arrays of double (C[i*n+j]).
 * Usage: ./matmul <input.bin> <n> <reps> <warmup> [output.bin]
 * Timer covers only the triple loop. File loading and allocation are outside.
 * Output (stdout): "rep,<idx>,<ns>" per measured run and "checksum,<sum of C>".
 */
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static long long now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

static void multiply(const double *A, const double *B, double *C, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++)
                sum += A[(size_t)i * n + k] * B[(size_t)k * n + j];
            C[(size_t)i * n + j] = sum;
        }
}

/* Peak resident set size of this process (Linux VmHWM, kB); -1 if unavailable. */
static long peak_rss_kb(void) {
    FILE *f = fopen("/proc/self/status", "r");
    char line[256];
    long v = -1;
    if (!f) return -1;
    while (fgets(line, sizeof line, f))
        if (sscanf(line, "VmHWM: %ld kB", &v) == 1) break;
    fclose(f);
    return v;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s input.bin n reps warmup [output.bin]\n", argv[0]);
        return 2;
    }
    int n = atoi(argv[2]), reps = atoi(argv[3]), warmup = atoi(argv[4]);
    size_t cnt = (size_t)n * n;
    double *A = malloc(cnt * sizeof(double));
    double *B = malloc(cnt * sizeof(double));
    double *C = calloc(cnt, sizeof(double));
    if (!A || !B || !C) { fprintf(stderr, "allocation failed\n"); return 3; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror("open input"); return 4; }
    if (fread(A, sizeof(double), cnt, f) != cnt || fread(B, sizeof(double), cnt, f) != cnt) {
        fprintf(stderr, "input file too short for n=%d\n", n);
        return 5;
    }
    fclose(f);

    for (int w = 0; w < warmup; w++) multiply(A, B, C, n);
    for (int r = 0; r < reps; r++) {
        long long t0 = now_ns();
        multiply(A, B, C, n);
        long long t1 = now_ns();
        printf("rep,%d,%lld\n", r, t1 - t0);
        fflush(stdout);
    }
    double checksum = 0.0;
    for (size_t i = 0; i < cnt; i++) checksum += C[i];
    printf("checksum,%.12e\n", checksum);
    printf("rss_kb,%ld\n", peak_rss_kb());

    if (argc > 5) {
        FILE *o = fopen(argv[5], "wb");
        fwrite(C, sizeof(double), cnt, o);
        fclose(o);
    }
    free(A); free(B); free(C);
    return 0;
}
