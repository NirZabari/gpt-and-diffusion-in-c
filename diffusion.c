/*
 * DDPM diffusion model from scratch in pure C.
 * Trains on MNIST images (28x28 grayscale) and writes generated samples as PNG.
 * Uses stb_image_write.h (single-header, public domain) for PNG output.
 *
 * Data file expected (download via Makefile target):
 *   train-images-idx3-ubyte
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ── Image / dataset ────────────────────────────────────────────── */
#define IMG_W 28
#define IMG_H 28
#define IMG_SIZE (IMG_W * IMG_H)

/* ── Diffusion hyperparameters ──────────────────────────────────── */
#define T_STEPS 1000
#define BETA_START 0.0001f
#define BETA_END 0.02f

/* ── Model hyperparameters (MLP denoiser) ───────────────────────── */
#define T_EMBD 64
#define INPUT_DIM (IMG_SIZE + T_EMBD)
#define H1 784
#define H2 784

/* ── Train / sample settings ────────────────────────────────────── */
#define TRAIN_STEPS 100000
#define PRINT_EVERY 100
#define SAMPLE_COUNT 20

#define GRID_N 16
#define UPSCALE 2
#define CELL_W (IMG_W * UPSCALE)
#define CELL_H (IMG_H * UPSCALE)
#define GRID_W (GRID_N * CELL_W)
#define GRID_H (GRID_N * CELL_H)

#define SNAPSHOT_EVERY 500
#define SNAP_N 8
#define SNAP_SEED 12345ULL
#define SNAP_W (SNAP_N * CELL_W)
#define SNAP_H (SNAP_N * CELL_H)

/* ================================================================
   RNG (xorshift64 + Box-Muller)
   ================================================================ */

static unsigned long long rng_state = 42;

static unsigned long long rng_next(void) {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 7;
  rng_state ^= rng_state << 17;
  return rng_state;
}

static double rng_uniform(void) {
  /* top 53 bits -> [0,1) */
  return (rng_next() >> 11) * (1.0 / 9007199254740992.0);
}

static float rng_gauss(float mean, float std) {
  double u1 = rng_uniform();
  double u2 = rng_uniform();
  if (u1 < 1e-30) u1 = 1e-30;
  return mean + std * (float)(sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2));
}

/* ================================================================
   MNIST loader (IDX3-UBYTE, big-endian header)
   ================================================================ */

static int read_be32(FILE *f) {
  unsigned char b[4];
  if (fread(b, 1, 4, f) != 4) return -1;
  return ((int)b[0] << 24) | ((int)b[1] << 16) | ((int)b[2] << 8) | (int)b[3];
}

static unsigned char *load_mnist_images(const char *path, int *out_count) {
  FILE *f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "Cannot open %s\n", path);
    return NULL;
  }

  int magic = read_be32(f);
  int count = read_be32(f);
  int rows = read_be32(f);
  int cols = read_be32(f);
  if (magic != 2051 || count <= 0 || rows != IMG_H || cols != IMG_W) {
    fprintf(stderr, "Bad MNIST header (magic=%d count=%d rows=%d cols=%d)\n",
            magic, count, rows, cols);
    fclose(f);
    return NULL;
  }

  size_t nbytes = (size_t)count * (size_t)rows * (size_t)cols;
  unsigned char *images = (unsigned char *)malloc(nbytes);
  if (!images) {
    fprintf(stderr, "OOM allocating MNIST (%zu bytes)\n", nbytes);
    fclose(f);
    return NULL;
  }

  if (fread(images, 1, nbytes, f) != nbytes) {
    fprintf(stderr, "Failed to read MNIST pixel data\n");
    free(images);
    fclose(f);
    return NULL;
  }
  fclose(f);

  *out_count = count;
  printf("Loaded %d MNIST images (%dx%d)\n", count, rows, cols);
  return images;
}

static void mnist_to_float(const unsigned char *img_u8, float *out_f32) {
  for (int i = 0; i < IMG_SIZE; i++) {
    float v = (float)img_u8[i] * (1.0f / 255.0f);
    out_f32[i] = v * 2.0f - 1.0f;
  }
}

/* ================================================================
   Diffusion schedule precompute
   ================================================================ */

static float betas[T_STEPS + 1];
static float alphas[T_STEPS + 1];
static float alpha_bars[T_STEPS + 1];
static float sqrt_alpha_bars[T_STEPS + 1];
static float sqrt_one_minus_alpha_bars[T_STEPS + 1];
static float posterior_var[T_STEPS + 1];
static float sqrt_posterior_var[T_STEPS + 1];

static void init_schedule(void) {
  alpha_bars[0] = 1.0f;
  double ab = 1.0;
  for (int t = 1; t <= T_STEPS; t++) {
    float frac = (float)(t - 1) / (float)(T_STEPS - 1);
    float beta = BETA_START + frac * (BETA_END - BETA_START);
    float alpha = 1.0f - beta;

    betas[t] = beta;
    alphas[t] = alpha;

    ab *= (double)alpha;
    alpha_bars[t] = (float)ab;
    sqrt_alpha_bars[t] = sqrtf(alpha_bars[t]);
    sqrt_one_minus_alpha_bars[t] = sqrtf(1.0f - alpha_bars[t]);

    if (t == 1) {
      posterior_var[t] = 0.0f;
      sqrt_posterior_var[t] = 0.0f;
    } else {
      /* beta_t * (1 - a_bar_{t-1}) / (1 - a_bar_t) */
      float num = beta * (1.0f - alpha_bars[t - 1]);
      float den = (1.0f - alpha_bars[t]);
      float pv = num / den;
      posterior_var[t] = pv;
      sqrt_posterior_var[t] = sqrtf(pv);
    }
  }
}

/* ================================================================
   Timestep embedding (sin/cos)
   ================================================================ */

static void timestep_embedding(int t, float *out) {
  /* Like transformer positional encoding, applied to timestep t. */
  const float base = 10000.0f;
  for (int i = 0; i < T_EMBD / 2; i++) {
    float exponent = -logf(base) * (2.0f * (float)i) / (float)T_EMBD;
    float inv_freq = expf(exponent);
    float arg = (float)t * inv_freq;
    out[2 * i] = sinf(arg);
    out[2 * i + 1] = cosf(arg);
  }
}

/* ================================================================
   MLP denoiser parameters
   ================================================================ */

static float *w1, *b1;
static float *w2, *b2;
static float *w3, *b3;

static float *dw1, *db1;
static float *dw2, *db2;
static float *dw3, *db3;

static float *mw1, *vw1, *mb1, *vb1;
static float *mw2, *vw2, *mb2, *vb2;
static float *mw3, *vw3, *mb3, *vb3;

static int num_params = 0;

static float *make_param(int n, float std) {
  float *p = (float *)calloc((size_t)n, sizeof(float));
  if (!p) return NULL;
  for (int i = 0; i < n; i++) p[i] = rng_gauss(0, std);
  num_params += n;
  return p;
}

static float *make_zero(int n) {
  float *p = (float *)calloc((size_t)n, sizeof(float));
  if (!p) return NULL;
  return p;
}

static void init_params(void) {
  const int w1_sz = H1 * INPUT_DIM;
  const int w2_sz = H2 * H1;
  const int w3_sz = IMG_SIZE * H2;

  w1 = make_param(w1_sz, 0.02f);
  b1 = make_zero(H1);
  w2 = make_param(w2_sz, 0.02f);
  b2 = make_zero(H2);
  w3 = make_param(w3_sz, 0.02f);
  b3 = make_zero(IMG_SIZE);

  dw1 = make_zero(w1_sz);
  db1 = make_zero(H1);
  dw2 = make_zero(w2_sz);
  db2 = make_zero(H2);
  dw3 = make_zero(w3_sz);
  db3 = make_zero(IMG_SIZE);

  mw1 = make_zero(w1_sz);
  vw1 = make_zero(w1_sz);
  mb1 = make_zero(H1);
  vb1 = make_zero(H1);

  mw2 = make_zero(w2_sz);
  vw2 = make_zero(w2_sz);
  mb2 = make_zero(H2);
  vb2 = make_zero(H2);

  mw3 = make_zero(w3_sz);
  vw3 = make_zero(w3_sz);
  mb3 = make_zero(IMG_SIZE);
  vb3 = make_zero(IMG_SIZE);

  if (!w1 || !b1 || !w2 || !b2 || !w3 || !b3 || !dw1 || !db1 || !dw2 || !db2 ||
      !dw3 || !db3 || !mw1 || !vw1 || !mb1 || !vb1 || !mw2 || !vw2 || !mb2 ||
      !vb2 || !mw3 || !vw3 || !mb3 || !vb3) {
    fprintf(stderr, "OOM initializing parameters\n");
    exit(1);
  }
}

/* ================================================================
   MLP forward/backward (ReLU)
   ================================================================ */

static inline float relu(float x) { return x > 0 ? x : 0; }

static void mlp_forward(const float *restrict in, float *restrict a1,
                        float *restrict a2, float *restrict out) {
  for (int r = 0; r < H1; r++) {
    const float *wr = w1 + (size_t)r * INPUT_DIM;
    float s = b1[r];
    for (int c = 0; c < INPUT_DIM; c++) s += wr[c] * in[c];
    a1[r] = relu(s);
  }

  for (int r = 0; r < H2; r++) {
    const float *wr = w2 + (size_t)r * H1;
    float s = b2[r];
    for (int c = 0; c < H1; c++) s += wr[c] * a1[c];
    a2[r] = relu(s);
  }

  for (int r = 0; r < IMG_SIZE; r++) {
    const float *wr = w3 + (size_t)r * H2;
    float s = b3[r];
    for (int c = 0; c < H2; c++) s += wr[c] * a2[c];
    out[r] = s;
  }
}

static void mlp_backward(const float *restrict in, const float *restrict a1,
                         const float *restrict a2,
                         const float *restrict dout) {
  float da2[H2];
  memset(da2, 0, sizeof(da2));

  for (int r = 0; r < IMG_SIZE; r++) {
    float dr = dout[r];
    db3[r] += dr;
    float *dwr = dw3 + (size_t)r * H2;
    for (int c = 0; c < H2; c++) dwr[c] += dr * a2[c];
  }

  for (int c = 0; c < H2; c++) {
    float s = 0;
    for (int r = 0; r < IMG_SIZE; r++) s += dout[r] * w3[(size_t)r * H2 + c];
    da2[c] = s;
  }

  for (int i = 0; i < H2; i++)
    if (a2[i] <= 0) da2[i] = 0;

  float da1[H1];
  memset(da1, 0, sizeof(da1));

  for (int r = 0; r < H2; r++) {
    float dr = da2[r];
    db2[r] += dr;
    float *dwr = dw2 + (size_t)r * H1;
    for (int c = 0; c < H1; c++) dwr[c] += dr * a1[c];
  }

  for (int c = 0; c < H1; c++) {
    float s = 0;
    for (int r = 0; r < H2; r++) s += da2[r] * w2[(size_t)r * H1 + c];
    da1[c] = s;
  }

  for (int i = 0; i < H1; i++)
    if (a1[i] <= 0) da1[i] = 0;

  for (int r = 0; r < H1; r++) {
    float dr = da1[r];
    db1[r] += dr;
    float *dwr = dw1 + (size_t)r * INPUT_DIM;
    for (int c = 0; c < INPUT_DIM; c++) dwr[c] += dr * in[c];
  }
}

/* ================================================================
   Adam optimizer
   ================================================================ */

static void adam_update(float *p, float *g, float *m, float *v, int sz, float lr,
                        float b1c, float b2c, float b1, float b2, float eps) {
  for (int i = 0; i < sz; i++) {
    m[i] = b1 * m[i] + (1 - b1) * g[i];
    v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
    p[i] -= lr * (m[i] / b1c) / (sqrtf(v[i] / b2c) + eps);
    g[i] = 0;
  }
}

static void update_all(int step, int total_steps, float lr) {
  /* cosine LR schedule */
  float lr_t =
      lr * 0.5f * (1.0f + cosf((float)M_PI * (float)step / (float)total_steps));

  const float beta1 = 0.9f;
  const float beta2 = 0.999f;
  const float eps = 1e-8f;

  float b1c = 1.0f - powf(beta1, step + 1);
  float b2c = 1.0f - powf(beta2, step + 1);

  const int w1_sz = H1 * INPUT_DIM;
  const int w2_sz = H2 * H1;
  const int w3_sz = IMG_SIZE * H2;

  adam_update(w1, dw1, mw1, vw1, w1_sz, lr_t, b1c, b2c, beta1, beta2, eps);
  adam_update(b1, db1, mb1, vb1, H1, lr_t, b1c, b2c, beta1, beta2, eps);
  adam_update(w2, dw2, mw2, vw2, w2_sz, lr_t, b1c, b2c, beta1, beta2, eps);
  adam_update(b2, db2, mb2, vb2, H2, lr_t, b1c, b2c, beta1, beta2, eps);
  adam_update(w3, dw3, mw3, vw3, w3_sz, lr_t, b1c, b2c, beta1, beta2, eps);
  adam_update(b3, db3, mb3, vb3, IMG_SIZE, lr_t, b1c, b2c, beta1, beta2, eps);
}

/* ================================================================
   PNG writer (via stb_image_write)
   ================================================================ */

static void write_png(const char *path, const float *img) {
  unsigned char buf[IMG_SIZE];
  for (int i = 0; i < IMG_SIZE; i++) {
    float v = (img[i] + 1.0f) * 0.5f;
    if (v < 0) v = 0;
    if (v > 1) v = 1;
    buf[i] = (unsigned char)(v * 255.0f + 0.5f);
  }
  if (!stbi_write_png(path, IMG_W, IMG_H, 1, buf, 0)) {
    fprintf(stderr, "Cannot write %s\n", path);
  }
}

/* ================================================================
   Training and sampling
   ================================================================ */

static void sample_xt(const float *x0, int t, float *xt, float *eps) {
  float sa = sqrt_alpha_bars[t];
  float so = sqrt_one_minus_alpha_bars[t];
  for (int i = 0; i < IMG_SIZE; i++) {
    float n = rng_gauss(0.0f, 1.0f);
    eps[i] = n;
    xt[i] = sa * x0[i] + so * n;
  }
}

static void ddpm_sample_to_buf(float *out) {
  for (int i = 0; i < IMG_SIZE; i++) out[i] = rng_gauss(0.0f, 1.0f);

  float t_emb[T_EMBD];
  float in[INPUT_DIM];
  float a1[H1], a2[H2], pred_x0[IMG_SIZE];

  for (int t = T_STEPS; t >= 1; t--) {
    timestep_embedding(t, t_emb);
    memcpy(in, out, IMG_SIZE * sizeof(float));
    memcpy(in + IMG_SIZE, t_emb, T_EMBD * sizeof(float));

    mlp_forward(in, a1, a2, pred_x0);

    float beta = betas[t];
    float alpha = alphas[t];
    float inv_sqrt_alpha = 1.0f / sqrtf(alpha);
    float denom = sqrt_one_minus_alpha_bars[t];
    if (denom < 1e-6f) denom = 1e-6f;
    float coef = beta / denom;

    float sigma = sqrt_posterior_var[t];
    for (int i = 0; i < IMG_SIZE; i++) {
      float eps_hat = (out[i] - sqrt_alpha_bars[t] * pred_x0[i]) / denom;
      float mean = inv_sqrt_alpha * (out[i] - coef * eps_hat);
      if (t > 1) mean += sigma * rng_gauss(0.0f, 1.0f);
      out[i] = mean;
    }
  }
}

static void write_grid_png(const char *path, int n) {
  int gw = n * CELL_W, gh = n * CELL_H;
  unsigned char *grid = (unsigned char *)calloc((size_t)gw * gh, 1);
  if (!grid) { fprintf(stderr, "OOM for grid\n"); return; }

  float sample[IMG_SIZE];
  int total = n * n;

  for (int idx = 0; idx < total; idx++) {
    ddpm_sample_to_buf(sample);
    printf("  grid sample %d / %d\r", idx + 1, total);
    fflush(stdout);

    int gr = idx / n;
    int gc = idx % n;
    int y0 = gr * CELL_H;
    int x0 = gc * CELL_W;

    for (int sy = 0; sy < IMG_H; sy++) {
      for (int sx = 0; sx < IMG_W; sx++) {
        float v = (sample[sy * IMG_W + sx] + 1.0f) * 0.5f;
        if (v < 0) v = 0;
        if (v > 1) v = 1;
        unsigned char pix = (unsigned char)(v * 255.0f + 0.5f);

        for (int dy = 0; dy < UPSCALE; dy++)
          for (int dx = 0; dx < UPSCALE; dx++)
            grid[(y0 + sy * UPSCALE + dy) * gw + (x0 + sx * UPSCALE + dx)] = pix;
      }
    }
  }
  printf("\n");

  if (!stbi_write_png(path, gw, gh, 1, grid, gw))
    fprintf(stderr, "Cannot write %s\n", path);
  else
    printf("Wrote %s (%dx%d)\n", path, gw, gh);

  free(grid);
}

static void write_snapshot_grid(const char *path) {
  unsigned char *grid = (unsigned char *)calloc((size_t)SNAP_W * SNAP_H, 1);
  if (!grid) { fprintf(stderr, "OOM for snapshot grid\n"); return; }

  unsigned long long saved_rng = rng_state;
  rng_state = SNAP_SEED;

  float sample[IMG_SIZE];
  int total = SNAP_N * SNAP_N;

  for (int idx = 0; idx < total; idx++) {
    ddpm_sample_to_buf(sample);

    int gr = idx / SNAP_N;
    int gc = idx % SNAP_N;
    int y0 = gr * CELL_H;
    int x0 = gc * CELL_W;

    for (int sy = 0; sy < IMG_H; sy++) {
      for (int sx = 0; sx < IMG_W; sx++) {
        float v = (sample[sy * IMG_W + sx] + 1.0f) * 0.5f;
        if (v < 0) v = 0;
        if (v > 1) v = 1;
        unsigned char pix = (unsigned char)(v * 255.0f + 0.5f);

        for (int dy = 0; dy < UPSCALE; dy++)
          for (int dx = 0; dx < UPSCALE; dx++)
            grid[(y0 + sy * UPSCALE + dy) * SNAP_W + (x0 + sx * UPSCALE + dx)] = pix;
      }
    }
  }

  if (!stbi_write_png(path, SNAP_W, SNAP_H, 1, grid, SNAP_W))
    fprintf(stderr, "Cannot write %s\n", path);
  else
    printf("Wrote snapshot %s\n", path);

  free(grid);
  rng_state = saved_rng;
}

static void free_all(void) {
  free(w1);
  free(b1);
  free(w2);
  free(b2);
  free(w3);
  free(b3);

  free(dw1);
  free(db1);
  free(dw2);
  free(db2);
  free(dw3);
  free(db3);

  free(mw1);
  free(vw1);
  free(mb1);
  free(vb1);
  free(mw2);
  free(vw2);
  free(mb2);
  free(vb2);
  free(mw3);
  free(vw3);
  free(mb3);
  free(vb3);
}

/* ================================================================
   Checkpoint save / load
   ================================================================ */

static void save_weights(const char *path) {
  FILE *f = fopen(path, "wb");
  if (!f) {
    fprintf(stderr, "Cannot write %s\n", path);
    return;
  }
  fwrite(w1, sizeof(float), H1 * INPUT_DIM, f);
  fwrite(b1, sizeof(float), H1, f);
  fwrite(w2, sizeof(float), H2 * H1, f);
  fwrite(b2, sizeof(float), H2, f);
  fwrite(w3, sizeof(float), IMG_SIZE * H2, f);
  fwrite(b3, sizeof(float), IMG_SIZE, f);
  fclose(f);
  printf("Saved weights to %s\n", path);
}

static int load_weights(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) return 0;
  int ok = 1;
  ok = ok && fread(w1, sizeof(float), H1 * INPUT_DIM, f) == H1 * INPUT_DIM;
  ok = ok && fread(b1, sizeof(float), H1, f) == H1;
  ok = ok && fread(w2, sizeof(float), H2 * H1, f) == H2 * H1;
  ok = ok && fread(b2, sizeof(float), H2, f) == H2;
  ok = ok && fread(w3, sizeof(float), IMG_SIZE * H2, f) == IMG_SIZE * H2;
  ok = ok && fread(b3, sizeof(float), IMG_SIZE, f) == IMG_SIZE;
  fclose(f);
  if (ok) printf("Loaded weights from %s\n", path);
  else fprintf(stderr, "Warning: incomplete read from %s\n", path);
  return ok;
}

int main(int argc, char **argv) {
  int sample_only = 0;
  float lr = 1e-3f;
  const char *outdir = "output/ddpm";
  const char *weights_path = "diffusion.bin";

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--sample") == 0) sample_only = 1;
    else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) lr = strtof(argv[++i], NULL);
    else if (strcmp(argv[i], "--outdir") == 0 && i + 1 < argc) outdir = argv[++i];
    else if (strcmp(argv[i], "--weights") == 0 && i + 1 < argc) weights_path = argv[++i];
  }

  char mkdir_cmd[512];
  snprintf(mkdir_cmd, sizeof(mkdir_cmd), "mkdir -p %s", outdir);

  init_schedule();
  init_params();

  if (sample_only) {
    if (!load_weights(weights_path)) {
      fprintf(stderr, "%s not found. Train first with: ./diffusion\n", weights_path);
      free_all();
      return 1;
    }
  } else {
    printf("num params: %d\n", num_params);
    printf("lr: %g  outdir: %s\n", lr, outdir);
    system(mkdir_cmd);

    char csv_path[512];
    snprintf(csv_path, sizeof(csv_path), "%s/loss.csv", outdir);
    FILE *csv = fopen(csv_path, "w");
    if (csv) fprintf(csv, "step,loss\n");

    int mnist_count = 0;
    unsigned char *mnist =
        load_mnist_images("train-images-idx3-ubyte", &mnist_count);
    if (!mnist) {
      fprintf(stderr, "MNIST data missing. Run: make data\n");
      if (csv) fclose(csv);
      free_all();
      return 1;
    }

    float run_loss = 0.0f;

    float x0[IMG_SIZE];
    float xt[IMG_SIZE];
    float eps[IMG_SIZE];
    float pred_x0[IMG_SIZE];
    float d_out[IMG_SIZE];

    float t_emb[T_EMBD];
    float in[INPUT_DIM];
    float a1[H1], a2[H2];

    for (int step = 0; step < TRAIN_STEPS; step++) {
      int idx = (int)(rng_uniform() * (double)mnist_count);
      int t = 1 + (int)(rng_uniform() * (double)T_STEPS);

      const unsigned char *img_u8 = mnist + (size_t)idx * IMG_SIZE;
      mnist_to_float(img_u8, x0);

      sample_xt(x0, t, xt, eps);
      timestep_embedding(t, t_emb);

      memcpy(in, xt, IMG_SIZE * sizeof(float));
      memcpy(in + IMG_SIZE, t_emb, T_EMBD * sizeof(float));

      mlp_forward(in, a1, a2, pred_x0);

      float loss = 0.0f;
      float inv = 2.0f / (float)IMG_SIZE;
      for (int i = 0; i < IMG_SIZE; i++) {
        float diff = pred_x0[i] - x0[i];
        loss += diff * diff;
        d_out[i] = inv * diff;
      }
      loss /= (float)IMG_SIZE;
      run_loss += loss;

      mlp_backward(in, a1, a2, d_out);
      update_all(step, TRAIN_STEPS, lr);

      if ((step + 1) % PRINT_EVERY == 0) {
        float avg = run_loss / (float)PRINT_EVERY;
        printf("step %6d / %6d | avg loss %.4f\n", step + 1, TRAIN_STEPS, avg);
        if (csv) { fprintf(csv, "%d,%.6f\n", step + 1, avg); fflush(csv); }
        run_loss = 0.0f;
      }

      if ((step + 1) % SNAPSHOT_EVERY == 0) {
        save_weights(weights_path);
        char snap_path[512];
        snprintf(snap_path, sizeof(snap_path),
                 "%s/grid_step_%06d.png", outdir, step + 1);
        printf("Snapshot at step %d...\n", step + 1);
        write_snapshot_grid(snap_path);
      }
    }

    save_weights(weights_path);
    if (csv) fclose(csv);
    free(mnist);
  }

  system(mkdir_cmd);

  printf("\nGenerating %d individual samples...\n", SAMPLE_COUNT);
  for (int i = 0; i < SAMPLE_COUNT; i++) {
    float x[IMG_SIZE];
    for (int j = 0; j < IMG_SIZE; j++) x[j] = rng_gauss(0.0f, 1.0f);

    float t_emb[T_EMBD];
    float in_buf[INPUT_DIM];
    float a1[H1], a2[H2], px0[IMG_SIZE];

    for (int t = T_STEPS; t >= 1; t--) {
      timestep_embedding(t, t_emb);
      memcpy(in_buf, x, IMG_SIZE * sizeof(float));
      memcpy(in_buf + IMG_SIZE, t_emb, T_EMBD * sizeof(float));
      mlp_forward(in_buf, a1, a2, px0);

      float beta = betas[t];
      float alpha = alphas[t];
      float inv_sqrt_alpha = 1.0f / sqrtf(alpha);
      float denom = sqrt_one_minus_alpha_bars[t];
      if (denom < 1e-6f) denom = 1e-6f;
      float coef = beta / denom;
      float sigma = sqrt_posterior_var[t];
      for (int j = 0; j < IMG_SIZE; j++) {
        float eps_hat = (x[j] - sqrt_alpha_bars[t] * px0[j]) / denom;
        float mean = inv_sqrt_alpha * (x[j] - coef * eps_hat);
        if (t > 1) mean += sigma * rng_gauss(0.0f, 1.0f);
        x[j] = mean;
      }
    }

    char path[512];
    snprintf(path, sizeof(path), "%s/sample_%02d.png", outdir, i + 1);
    write_png(path, x);
    printf("Wrote %s\n", path);
  }

  printf("\nGenerating 10x10 grid...\n");
  { char p[512]; snprintf(p, sizeof(p), "%s/grid_10x10.png", outdir); write_grid_png(p, 10); }
  printf("\nGenerating 20x20 grid...\n");
  { char p[512]; snprintf(p, sizeof(p), "%s/grid_20x20.png", outdir); write_grid_png(p, 20); }

  free_all();
  return 0;
}

