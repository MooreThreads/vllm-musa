// SPDX-License-Identifier: Apache-2.0
// Two byte all-gathers, optionally grouped, with a bounded stream wait.
#define _POSIX_C_SOURCE 200809L
#include <musa_runtime.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

// MCCL 2.30.4 ABI subset for images that omit mccl.h.
typedef struct mcclComm* mcclComm_t;
typedef struct { char internal[128]; } mcclUniqueId;
extern int mcclGetVersion(int*);
extern int mcclGetUniqueId(mcclUniqueId*);
extern int mcclCommInitRank(mcclComm_t*, int, mcclUniqueId, int);
extern int mcclCommDestroy(mcclComm_t);
extern int mcclAllGather(const void*, void*, size_t, int, mcclComm_t, musaStream_t);
extern int mcclGroupStart(void);
extern int mcclGroupEnd(void);

static volatile sig_atomic_t waiting_for_stream;

static void timed_out(int signal_number) {
  (void)signal_number;
  if (waiting_for_stream) {
    static const char message[] = "TIMEOUT: group end returned; GPU stream did not finish\n";
    (void)!write(STDERR_FILENO, message, sizeof(message) - 1);
  } else {
    static const char message[] = "TIMEOUT: initialization or enqueue did not finish\n";
    (void)!write(STDERR_FILENO, message, sizeof(message) - 1);
  }
  _exit(124);
}

#define MC(call) do { int code = (call); if (code) { \
  fprintf(stderr, "MCCL error %d: %s\n", code, #call); exit(1); } } while (0)
#define MU(call) do { musaError_t code = (call); if (code != musaSuccess) { \
  fprintf(stderr, "MUSA error %s: %s\n", musaGetErrorString(code), #call); exit(1); } } while (0)

int main(int argc, char** argv) {
  if (argc != 7) {
    fprintf(stderr, "usage: %s RANK WORLD ID_FILE group|nogroup LARGE_BYTES SMALL_BYTES\n", argv[0]);
    return 2;
  }
  int rank = atoi(argv[1]), world = atoi(argv[2]);
  int grouped = !strcmp(argv[4], "group");
  size_t bytes[] = {strtoul(argv[5], NULL, 10), strtoul(argv[6], NULL, 10)};
  if (world < 2 || world > 8 || rank < 0 || rank >= world ||
      (!grouped && strcmp(argv[4], "nogroup")) ||
      !bytes[0] || !bytes[1] || bytes[0] > (1U << 20) || bytes[1] > (1U << 20)) return 2;
  struct sigaction action = {0};
  action.sa_handler = timed_out;
  sigemptyset(&action.sa_mask);
  if (sigaction(SIGALRM, &action, NULL)) return 1;
  alarm(90);
  MU(musaSetDevice(rank));
  mcclUniqueId id;
  if (!rank) {
    MC(mcclGetUniqueId(&id));
    char temporary[1024];
    if (snprintf(temporary, sizeof(temporary), "%s.tmp", argv[3]) >= (int)sizeof(temporary)) return 2;
    FILE* file = fopen(temporary, "wb");
    if (!file || fwrite(&id, sizeof(id), 1, file) != 1 || fclose(file)) return 1;
    if (rename(temporary, argv[3])) return 1;
  } else {
    for (;;) {
      FILE* file = fopen(argv[3], "rb");
      if (file) {
        size_t count = fread(&id, sizeof(id), 1, file);
        fclose(file);
        if (count == 1) break;
      }
      struct timespec delay = {.tv_nsec = 10000000};
      nanosleep(&delay, NULL);
    }
  }
  mcclComm_t comm;
  MC(mcclCommInitRank(&comm, world, id, rank));
  musaStream_t stream;
  MU(musaStreamCreate(&stream));
  void *send[2], *receive[2];
  for (int i = 0; i < 2; ++i) {
    MU(musaMalloc(&send[i], bytes[i]));
    MU(musaMalloc(&receive[i], bytes[i] * (size_t)world));
    MU(musaMemset(send[i], rank + 1, bytes[i]));
    MU(musaMemset(receive[i], 0, bytes[i] * (size_t)world));
  }
  MU(musaDeviceSynchronize());
  alarm(20);
  if (grouped) MC(mcclGroupStart());
  for (int i = 0; i < 2; ++i) {
    // mcclUint8=1: use bytes so the result is independent of FP8 conversion.
    MC(mcclAllGather(send[i], receive[i], bytes[i], 1, comm, stream));
  }
  if (grouped) MC(mcclGroupEnd());
  waiting_for_stream = 1;
  MU(musaStreamSynchronize(stream));
  waiting_for_stream = 0;
  alarm(20);
  for (int i = 0; i < 2; ++i) {
    uint8_t* host = malloc(bytes[i] * (size_t)world);
    if (!host) return 1;
    MU(musaMemcpy(host, receive[i], bytes[i] * (size_t)world, musaMemcpyDeviceToHost));
    for (int peer = 0; peer < world; ++peer) {
      for (size_t offset = 0; offset < bytes[i]; ++offset) {
        if (host[(size_t)peer * bytes[i] + offset] != peer + 1) return 1;
      }
    }
    free(host);
    MU(musaFree(send[i]));
    MU(musaFree(receive[i]));
  }
  MC(mcclCommDestroy(comm));
  MU(musaStreamDestroy(stream));
  alarm(0);
  int version = 0;
  MC(mcclGetVersion(&version));
  printf("PASS rank=%d world=%d mode=%s bytes=%zu,%zu mccl=%d\n", rank, world, argv[4], bytes[0], bytes[1], version);
  return 0;
}
