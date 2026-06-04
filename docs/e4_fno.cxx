/**
 * E4 — FNO vs IgANet (E1): Representation Ablation
 *
 * Same Poisson problem, same data, same information content.
 * Only difference: spline coefficients (E1/IgANet) vs grid field (E4/FNO).
 *
 * E4a: Accuracy comparison       — full 400 training samples
 * E4b: Data efficiency           — N ∈ {50, 100, 200, 400}
 * E4c: Inference speed           — 1000 forward passes each
 *
 * FNO config: in_ch=2, modes=12, width=20, layers=4 (parametric domain)
 * E1  config: 50 → 256 → 256 → 100 (spline coefficients)
 */

#include <iganet.h>
#include <solver/ezsolver.hpp>
#include <chrono>
#include <iomanip>

using namespace iganet;
using Geo = S<UniformBSpline<double, 2, 2, 2>>;
using Var = S<UniformBSpline<double, 1, 3, 3>>;

// ═══════════════════════════════════════════════════════════════════════════
// FNO modules
// ═══════════════════════════════════════════════════════════════════════════

struct SpectralConv2dImpl : torch::nn::Module {
  int64_t modes1, modes2;
  torch::Tensor w1, w2; // [in_ch, out_ch, modes1, modes2, 2]  (real-as-complex)

  SpectralConv2dImpl(int64_t in_ch, int64_t out_ch, int64_t m1, int64_t m2)
      : modes1(m1), modes2(m2) {
    double s = 1.0 / (in_ch * out_ch);
    w1 = register_parameter("w1", s * torch::rand({in_ch, out_ch, m1, m2, 2}));
    w2 = register_parameter("w2", s * torch::rand({in_ch, out_ch, m1, m2, 2}));
  }

  // Spectral matmul: [B, in_ch, m1, m2] × [in_ch, out_ch, m1, m2] → [B, out_ch, m1, m2]
  torch::Tensor cmul(torch::Tensor x, torch::Tensor w) {
    return torch::einsum("bixy,ioxy->boxy", {x, w});
  }

  torch::Tensor forward(torch::Tensor x) {
    // x: [B, C, H, W]
    int64_t H = x.size(2), W = x.size(3);
    int64_t out_ch = w1.size(1);

    auto x_ft = torch::fft_rfft2(x, c10::nullopt, {-2LL, -1LL}); // [B,C,H,W/2+1] complex
    int64_t W_ft = W / 2 + 1;

    auto wc1 = torch::view_as_complex(w1); // [in_ch, out_ch, m1, m2] complex
    auto wc2 = torch::view_as_complex(w2);

    auto out_ft = torch::zeros({x.size(0), out_ch, H, W_ft}, x_ft.options());

    using S = torch::indexing::Slice;
    // Lower block (low positive frequencies)
    out_ft.index_put_({S(), S(), S(0, modes1), S(0, modes2)},
        cmul(x_ft.index({S(), S(), S(0, modes1), S(0, modes2)}), wc1));
    // Upper block (low negative frequencies)
    out_ft.index_put_({S(), S(), S(H - modes1, H), S(0, modes2)},
        cmul(x_ft.index({S(), S(), S(H - modes1, H), S(0, modes2)}), wc2));

    return torch::fft_irfft2(out_ft, {H, W}, {-2LL, -1LL}); // [B, out_ch, H, W]
  }
};
TORCH_MODULE(SpectralConv2d);

struct FNO2dImpl : torch::nn::Module {
  static constexpr int64_t NL = 4;
  torch::nn::Linear        fc0{nullptr}, fc1{nullptr}, fc2{nullptr};
  SpectralConv2d           spec[NL]{nullptr, nullptr, nullptr, nullptr};
  torch::nn::Conv2d        bypass[NL]{nullptr, nullptr, nullptr, nullptr};

  FNO2dImpl(int64_t in_ch, int64_t modes, int64_t width) {
    fc0 = register_module("fc0", torch::nn::Linear(in_ch, width));
    for (int64_t i = 0; i < NL; ++i) {
      spec[i]   = register_module("spec"   + std::to_string(i),
                    SpectralConv2d(width, width, modes, modes));
      bypass[i] = register_module("bypass" + std::to_string(i),
                    torch::nn::Conv2d(torch::nn::Conv2dOptions(width, width, 1)));
    }
    fc1 = register_module("fc1", torch::nn::Linear(width, 128));
    fc2 = register_module("fc2", torch::nn::Linear(128, 1));
  }

  // x: [B, H, W, in_ch] → [B, H, W]
  torch::Tensor forward(torch::Tensor x) {
    x = torch::gelu(fc0->forward(x)).permute({0, 3, 1, 2}); // [B, width, H, W]
    for (int64_t i = 0; i < NL; ++i)
      x = torch::gelu(spec[i]->forward(x) + bypass[i]->forward(x));
    x = x.permute({0, 2, 3, 1});            // [B, H, W, width]
    x = torch::gelu(fc1->forward(x));        // [B, H, W, 128]
    return fc2->forward(x).squeeze(-1);      // [B, H, W]
  }

  int64_t nparams() {
    int64_t n = 0;
    for (auto& p : parameters()) n += p.numel();
    return n;
  }
};
TORCH_MODULE(FNO2d);

// ═══════════════════════════════════════════════════════════════════════════
// E1 / IgANet
// ═══════════════════════════════════════════════════════════════════════════

struct PoissonNet
    : public IgANet<torch::optim::Adam, std::tuple<Geo>, std::tuple<Var>> {

  using Base = IgANet<torch::optim::Adam, std::tuple<Geo>, std::tuple<Var>>;
  torch::Tensor G_data, u_ref, cur_u_ref;

  PoissonNet(torch::Tensor G_, torch::Tensor u_)
      : Base({256, 256},
             {{{activation::tanh}}, {{activation::tanh}}, {{activation::none}}},
             std::tuple{std::array<int64_t, 2>{5, 5}},
             std::tuple{std::array<int64_t, 2>{10, 10}},
             init::greville),
        G_data(std::move(G_)), u_ref(std::move(u_)) {}

  bool epoch(int64_t) override {
    int64_t idx = torch::randint(G_data.size(0), {1}).item<int64_t>();
    this->input<0>().from_tensor(G_data[idx]);
    cur_u_ref = u_ref[idx];
    return true;
  }
  torch::Tensor loss(const torch::Tensor& out, int64_t) override {
    return torch::mse_loss(out, cur_u_ref);
  }
  int64_t nparams() {
    int64_t n = 0;
    for (auto& p : parameters()) n += p.numel();
    return n;
  }
};

// ═══════════════════════════════════════════════════════════════════════════
// FNO training loop (1 sample / step, matches E1 setup)
// ═══════════════════════════════════════════════════════════════════════════

void train_fno(FNO2d& fno,
               const torch::Tensor& G_grid_train, // [N, 64, 64, 2]
               const torch::Tensor& u_grid_train, // [N, 64, 64]
               int64_t max_epochs) {
  torch::optim::Adam opt(fno->parameters(), torch::optim::AdamOptions(1e-3));
  int64_t N = G_grid_train.size(0);
  double prev = -1.0;

  for (int64_t ep = 0; ep < max_epochs; ++ep) {
    int64_t idx = torch::randint(N, {1}).item<int64_t>();
    auto x = G_grid_train[idx].unsqueeze(0); // [1, 64, 64, 2]
    auto y = u_grid_train[idx].unsqueeze(0); // [1, 64, 64]

    fno->zero_grad();
    auto loss = torch::mse_loss(fno->forward(x), y);
    loss.backward();
    opt.step();

    double cur = loss.item<double>();
    if (cur < 1e-7 ||
        (prev > 0 && std::abs(cur - prev) / (cur + 1e-12) < 1e-6)) {
      Log(log::info) << "FNO converged at epoch " << ep
                     << ", loss=" << cur << "\n";
      return;
    }
    prev = cur;
  }
  Log(log::info) << "FNO max epochs reached, loss=" << prev << "\n";
}

// ═══════════════════════════════════════════════════════════════════════════
// Parametric grid (shared across evaluation calls)
// ═══════════════════════════════════════════════════════════════════════════

utils::TensorArray<2> make_xi_grid(int64_t Ngrid) {
  auto xi_1d = torch::linspace(0.0, 1.0, Ngrid,
                               torch::TensorOptions().dtype(torch::kDouble));
  utils::TensorArray<2> xi;
  xi[0] = xi_1d.repeat_interleave(Ngrid);
  xi[1] = xi_1d.repeat(Ngrid);
  return xi;
}

// ═══════════════════════════════════════════════════════════════════════════
// Evaluation helpers
// ═══════════════════════════════════════════════════════════════════════════

// E1: relative L2 error in field space (project coefficients → 64×64 grid)
double eval_e1_field(PoissonNet& net,
                     const torch::Tensor& G_test,      // [N, 50]
                     const torch::Tensor& u_grid_test, // [N, 64, 64]
                     const utils::TensorArray<2>& xi_grid,
                     int64_t Ngrid) {
  int64_t N = G_test.size(0);
  double sum = 0.0;
  for (int64_t i = 0; i < N; ++i) {
    net.input<0>().from_tensor(G_test[i]);
    net.eval();
    // Project predicted coefficients to grid
    auto vals    = net.output<0>().space<0>().eval(xi_grid); // BlockTensor<T,1,1>
    auto u_field = vals[0]->reshape({Ngrid, Ngrid});         // [64, 64]
    auto u_ref   = u_grid_test[i];
    double err = (u_field - u_ref).norm().item<double>();
    double ref = u_ref.norm().item<double>();
    sum += (ref > 0 ? err / ref : err);
  }
  return sum / N;
}

// FNO: relative L2 error in field space
double eval_fno_field(FNO2d& fno,
                      const torch::Tensor& G_grid_test, // [N, 64, 64, 2]
                      const torch::Tensor& u_grid_test, // [N, 64, 64]
                      int64_t N) {
  double sum = 0.0;
  for (int64_t i = 0; i < N; ++i) {
    auto u_pred = fno->forward(G_grid_test[i].unsqueeze(0)).squeeze(0); // [64,64]
    auto u_ref  = u_grid_test[i];
    double err = (u_pred - u_ref).norm().item<double>();
    double ref = u_ref.norm().item<double>();
    sum += (ref > 0 ? err / ref : err);
  }
  return sum / N;
}

// ═══════════════════════════════════════════════════════════════════════════
// E4a — Accuracy comparison (full 400 training samples)
// ═══════════════════════════════════════════════════════════════════════════

std::pair<PoissonNet, FNO2d>
run_e4a(const torch::Tensor& G_train,       const torch::Tensor& u_train,
        const torch::Tensor& G_grid_train,  const torch::Tensor& u_grid_train,
        const torch::Tensor& G_test,        const torch::Tensor& u_test,
        const torch::Tensor& G_grid_test,   const torch::Tensor& u_grid_test,
        const utils::TensorArray<2>& xi_grid, int64_t Ngrid) {

  Log() << "\n══ E4a: Accuracy comparison (N_train=" << G_train.size(0) << ") ══\n";

  // Train E1
  PoissonNet e1(G_train, u_train);
  e1.options().max_epoch(10000).min_loss(1e-7).min_loss_rel_change(1e-6);
  e1.optimizerReset(torch::optim::AdamOptions(1e-3));
  Log() << "[E4a] Training E1 (" << e1.nparams() << " params)...\n";
  e1.train();

  // Train FNO
  FNO2d fno(2, 12, 20);
  fno->to(torch::kDouble);
  Log() << "[E4a] Training FNO (" << fno->nparams() << " params)...\n";
  train_fno(fno, G_grid_train, u_grid_train, 10000);

  // Evaluate
  double e1_l2  = eval_e1_field(e1,  G_test,      u_grid_test, xi_grid, Ngrid);
  double fno_l2 = eval_fno_field(fno, G_grid_test, u_grid_test, u_grid_test.size(0));

  // E1 coefficient-space error (same as in E1 paper)
  {
    double sum = 0;
    int64_t N = G_test.size(0);
    for (int64_t i = 0; i < N; ++i) {
      e1.input<0>().from_tensor(G_test[i]);
      e1.eval();
      auto u_pred = e1.output<0>().as_tensor();
      double err = (u_pred - u_test[i]).norm().item<double>();
      double ref = u_test[i].norm().item<double>();
      sum += (ref > 0 ? err / ref : err);
    }
    Log() << "[E4a] E1  coeff-space rel-L2 : " << sum / N << "\n";
  }

  Log() << "[E4a] E1  field-space rel-L2 : " << e1_l2  << "\n";
  Log() << "[E4a] FNO field-space rel-L2 : " << fno_l2 << "\n";
  Log() << "[E4a] E1  params             : " << e1.nparams()     << "\n";
  Log() << "[E4a] FNO params             : " << fno->nparams()   << "\n";

  return {std::move(e1), std::move(fno)};
}

// ═══════════════════════════════════════════════════════════════════════════
// E4b — Data efficiency  N ∈ {50, 100, 200, 400}
// ═══════════════════════════════════════════════════════════════════════════

void run_e4b(const torch::Tensor& G_all,       const torch::Tensor& u_all,
             const torch::Tensor& G_grid_all,  const torch::Tensor& u_grid_all,
             const torch::Tensor& G_test,
             const torch::Tensor& G_grid_test, const torch::Tensor& u_grid_test,
             const utils::TensorArray<2>& xi_grid, int64_t Ngrid) {

  Log() << "\n══ E4b: Data efficiency ══\n";
  Log() << "[E4b] N_train   E1_field_L2   FNO_field_L2\n";
  Log() << "[E4b] --------  -----------   ------------\n";

  for (int64_t Ntrain : {50, 100, 200, 400}) {
    auto G_tr      = G_all.slice(0, 0, Ntrain).contiguous();
    auto u_tr      = u_all.slice(0, 0, Ntrain).contiguous();
    auto Gg_tr     = G_grid_all.slice(0, 0, Ntrain).contiguous();
    auto ug_tr     = u_grid_all.slice(0, 0, Ntrain).contiguous();

    // Train E1
    PoissonNet e1(G_tr, u_tr);
    e1.options().max_epoch(5000).min_loss(1e-7).min_loss_rel_change(1e-6);
    e1.optimizerReset(torch::optim::AdamOptions(1e-3));
    e1.train();

    // Train FNO
    FNO2d fno(2, 12, 20);
    fno->to(torch::kDouble);
    train_fno(fno, Gg_tr, ug_tr, 5000);

    double e1_l2  = eval_e1_field(e1,  G_test,      u_grid_test, xi_grid, Ngrid);
    double fno_l2 = eval_fno_field(fno, G_grid_test, u_grid_test, G_test.size(0));

    Log() << "[E4b] " << std::setw(8) << Ntrain
          << "  " << std::fixed << std::setprecision(6) << e1_l2
          << "   " << fno_l2 << "\n";
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// E4c — Inference speed
// ═══════════════════════════════════════════════════════════════════════════

void run_e4c(PoissonNet& e1, FNO2d& fno,
             const torch::Tensor& G_test,
             const torch::Tensor& G_grid_test) {
  Log() << "\n══ E4c: Inference speed (1000 forward passes) ══\n";
  int64_t NREP = 1000;
  int64_t N    = G_test.size(0);

  // Warm up
  for (int i = 0; i < 10; ++i) {
    e1.input<0>().from_tensor(G_test[i % N]);
    e1.eval();
    fno->forward(G_grid_test[i % N].unsqueeze(0));
  }

  // Time E1
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int64_t i = 0; i < NREP; ++i) {
    e1.input<0>().from_tensor(G_test[i % N]);
    e1.eval();
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  double e1_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / NREP;

  // Time FNO
  auto t2 = std::chrono::high_resolution_clock::now();
  for (int64_t i = 0; i < NREP; ++i)
    fno->forward(G_grid_test[i % N].unsqueeze(0));
  auto t3 = std::chrono::high_resolution_clock::now();
  double fno_ms = std::chrono::duration<double, std::milli>(t3 - t2).count() / NREP;

  Log() << "[E4c] E1  mean inference : " << std::fixed << std::setprecision(4)
        << e1_ms  << " ms / sample\n";
  Log() << "[E4c] FNO mean inference : " << fno_ms << " ms / sample\n";
  Log() << "[E4c] Speedup E1 / FNO  : " << fno_ms / e1_ms << "x\n";
}

// ═══════════════════════════════════════════════════════════════════════════
// main
// ═══════════════════════════════════════════════════════════════════════════

int main() {
  init();

  // Load dataset  [0]=G [1]=u [2]=u_val [3]=A [4]=b [5]=G_grid [6]=u_grid
  std::vector<torch::Tensor> tensors;
  torch::load(tensors, "dataset.pt");
  auto G_all      = tensors[0].to(torch::kDouble); // [500, 50]
  auto u_all      = tensors[1].to(torch::kDouble); // [500, 100]
  auto G_grid_all = tensors[5].to(torch::kDouble); // [500, 64, 64, 2]
  auto u_grid_all = tensors[6].to(torch::kDouble); // [500, 64, 64]

  int64_t N       = G_all.size(0);
  int64_t N_train = static_cast<int64_t>(N * 0.8); // 400
  int64_t N_test  = N - N_train;                    // 100
  constexpr int64_t Ngrid = 64;

  auto G_train      = G_all.slice(0, 0, N_train).contiguous();
  auto u_train      = u_all.slice(0, 0, N_train).contiguous();
  auto G_grid_train = G_grid_all.slice(0, 0, N_train).contiguous();
  auto u_grid_train = u_grid_all.slice(0, 0, N_train).contiguous();
  auto G_test       = G_all.slice(0, N_train).contiguous();
  auto u_test       = u_all.slice(0, N_train).contiguous();
  auto G_grid_test  = G_grid_all.slice(0, N_train).contiguous();
  auto u_grid_test  = u_grid_all.slice(0, N_train).contiguous();

  auto xi_grid = make_xi_grid(Ngrid);

  Log() << "[E4] Dataset: " << N << "  (train=" << N_train
        << ", test=" << N_test << ", grid=" << Ngrid << "×" << Ngrid << ")\n";

  // E4a — accuracy (also returns trained models for E4c)
  auto [e1, fno] = run_e4a(
      G_train, u_train, G_grid_train, u_grid_train,
      G_test,  u_test,  G_grid_test,  u_grid_test,
      xi_grid, Ngrid);

  // E4b — data efficiency
  run_e4b(G_all, u_all, G_grid_all, u_grid_all,
          G_test, G_grid_test, u_grid_test, xi_grid, Ngrid);

  // E4c — inference speed
  run_e4c(e1, fno, G_test, G_grid_test);

  finalize();
  return 0;
}
