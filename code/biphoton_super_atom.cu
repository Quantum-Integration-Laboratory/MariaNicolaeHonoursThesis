// compile: nvcc biphoton_super_atom.cu -o sim -O3 -rdc=true -lm -arch=sm_60
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define PI 3.1415926535897932384626433832795
const double HBAR = 1.054571817e-34;
const double K_B = 1.380649e-23;

#define NUM_THREAD_BLOCKS 256
#define NUM_THREADS_IN_BLOCK 512

enum SystemType
{
    LAMBDA_SYSTEM,
    V_SYSTEM
};
typedef enum SystemType SystemType;

struct DensityMatrix
{
    double r11;
    double r22;
    double r33;
    double r12;
    double i12;
    double r13;
    double i13;
    double r23;
    double i23;
};
typedef struct DensityMatrix DensityMatrix;

const DensityMatrix GROUND_STATE_MATRIX = {
    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
};

struct AtomSample
{
    double weight;
    double g_or;
    double g_oi;
    double g_mur;
    double g_mui;
    double delta_12;
    double delta_23;
};
typedef struct AtomSample AtomSample;

struct SystemState
{
    double alpha_r;
    double alpha_i;
    double beta_r;
    double beta_i;
    DensityMatrix rho[];
};
typedef struct SystemState SystemState;

size_t SizeofStateStruct(size_t nAtomSamples)
{
    return sizeof(SystemState) + nAtomSamples * sizeof(DensityMatrix);
}

bool HostAllocateStateStruct(SystemState **ptr, size_t nAtomSamples)
{
    *ptr = (SystemState *) malloc(SizeofStateStruct(nAtomSamples));
    return (*ptr != NULL);
}

bool DeviceAllocateStateStruct(SystemState **ptr, size_t nAtomSamples)
{
    cudaError_t result = cudaMalloc(ptr, SizeofStateStruct(nAtomSamples));
    return (result == cudaSuccess);
}

void CopyStateStruct(SystemState *dst, const SystemState *src,
                     size_t nAtomSamples, cudaMemcpyKind kind)
{
    cudaMemcpy(dst, src, SizeofStateStruct(nAtomSamples), kind);
}

bool HostAllocateAtomSamples(AtomSample **ptr, size_t nAtomSamples)
{
    size_t size = nAtomSamples * sizeof(AtomSample);
    *ptr = (AtomSample *) malloc(size);
    return (*ptr != NULL);
}

bool DeviceAllocateAtomSamples(AtomSample **ptr, size_t nAtomSamples)
{
    size_t size = nAtomSamples * sizeof(AtomSample);
    cudaError_t result = cudaMalloc(ptr, size);
    return (result == cudaSuccess);
}

void CopyAtomSamples(AtomSample *dst, const AtomSample *src,
                     size_t nAtomSamples, cudaMemcpyKind kind)
{
    cudaMemcpy(dst, src, nAtomSamples * sizeof(AtomSample), kind);
}

__device__
void MasterDerivative(DensityMatrix *diff, SystemType sys,
    const DensityMatrix *rho, double Omega_mur, double Omega_mui,
    double Omega_or, double Omega_oi, double Omega_pr, double Omega_pi,
    double deltap_mu, double deltap_o, double n_b, double gamma_12,
    double gamma_13, double gamma_23, double gamma_2d, double gamma_3d)
{
    DensityMatrix tmp = *rho;

    // code generated using computer algebra
    if (sys == LAMBDA_SYSTEM)
    {
        diff->r11 = -2*Omega_mui*tmp.r12 - 2*Omega_mur*tmp.i12 - 2*Omega_pi*tmp.r13 - 2*Omega_pr*tmp.i13 - gamma_12*n_b*tmp.r11 + gamma_12*tmp.r22*(n_b + 1) + gamma_13*tmp.r33;
        diff->r22 = 2*Omega_mui*tmp.r12 + 2*Omega_mur*tmp.i12 - 2*Omega_oi*tmp.r23 - 2*Omega_or*tmp.i23 + gamma_12*n_b*tmp.r11 - gamma_12*tmp.r22*(n_b + 1) + gamma_23*tmp.r33;
        diff->r33 = 2*Omega_oi*tmp.r23 + 2*Omega_or*tmp.i23 + 2*Omega_pi*tmp.r13 + 2*Omega_pr*tmp.i13 - gamma_13*tmp.r33 - gamma_23*tmp.r33;
        diff->r12 = Omega_mui*tmp.r11 - Omega_mui*tmp.r22 - Omega_oi*tmp.r13 - Omega_or*tmp.i13 - Omega_pi*tmp.r23 - Omega_pr*tmp.i23 - deltap_mu*tmp.i12 - 1.0/2.0*gamma_12*n_b*tmp.r12 - 1.0/2.0*gamma_12*tmp.r12*(n_b + 1) - 1.0/2.0*gamma_2d*tmp.r12;
        diff->i12 = Omega_mur*tmp.r11 - Omega_mur*tmp.r22 - Omega_oi*tmp.i13 + Omega_or*tmp.r13 + Omega_pi*tmp.i23 - Omega_pr*tmp.r23 + deltap_mu*tmp.r12 - 1.0/2.0*gamma_12*n_b*tmp.i12 - 1.0/2.0*gamma_12*tmp.i12*(n_b + 1) - 1.0/2.0*gamma_2d*tmp.i12;
        diff->r13 = -Omega_mui*tmp.r23 + Omega_mur*tmp.i23 + Omega_oi*tmp.r12 - Omega_or*tmp.i12 + Omega_pi*tmp.r11 - Omega_pi*tmp.r33 - 1.0/2.0*gamma_12*n_b*tmp.r13 - 1.0/2.0*gamma_13*tmp.r13 - 1.0/2.0*gamma_23*tmp.r13 - 1.0/2.0*gamma_3d*tmp.r13 + tmp.i13*(-deltap_mu - deltap_o);
        diff->i13 = -Omega_mui*tmp.i23 - Omega_mur*tmp.r23 + Omega_oi*tmp.i12 + Omega_or*tmp.r12 + Omega_pr*tmp.r11 - Omega_pr*tmp.r33 - 1.0/2.0*gamma_12*n_b*tmp.i13 - 1.0/2.0*gamma_13*tmp.i13 - 1.0/2.0*gamma_23*tmp.i13 - 1.0/2.0*gamma_3d*tmp.i13 - tmp.r13*(-deltap_mu - deltap_o);
        diff->r23 = Omega_mui*tmp.r13 + Omega_mur*tmp.i13 + Omega_oi*tmp.r22 - Omega_oi*tmp.r33 + Omega_pi*tmp.r12 + Omega_pr*tmp.i12 + deltap_mu*tmp.i23 - 1.0/2.0*gamma_12*tmp.r23*(n_b + 1) - 1.0/2.0*gamma_13*tmp.r23 - 1.0/2.0*gamma_23*tmp.r23 - 1.0/2.0*gamma_2d*tmp.r23 - 1.0/2.0*gamma_3d*tmp.r23 + tmp.i23*(-deltap_mu - deltap_o);
        diff->i23 = Omega_mui*tmp.i13 - Omega_mur*tmp.r13 + Omega_or*tmp.r22 - Omega_or*tmp.r33 - Omega_pi*tmp.i12 + Omega_pr*tmp.r12 - deltap_mu*tmp.r23 - 1.0/2.0*gamma_12*tmp.i23*(n_b + 1) - 1.0/2.0*gamma_13*tmp.i23 - 1.0/2.0*gamma_23*tmp.i23 - 1.0/2.0*gamma_2d*tmp.i23 - 1.0/2.0*gamma_3d*tmp.i23 - tmp.r23*(-deltap_mu - deltap_o);
    }
    else // sys == V_SYSTEM
    {
        diff->r11 = -2*Omega_oi*tmp.r12 - 2*Omega_or*tmp.i12 - 2*Omega_pi*tmp.r13 - 2*Omega_pr*tmp.i13 + gamma_12*tmp.r22 + gamma_13*tmp.r33;
        diff->r22 = -2*Omega_mui*tmp.r23 - 2*Omega_mur*tmp.i23 + 2*Omega_oi*tmp.r12 + 2*Omega_or*tmp.i12 - gamma_12*tmp.r22 - gamma_23*n_b*tmp.r22 + gamma_23*tmp.r33*(n_b + 1);
        diff->r33 = 2*Omega_mui*tmp.r23 + 2*Omega_mur*tmp.i23 + 2*Omega_pi*tmp.r13 + 2*Omega_pr*tmp.i13 - gamma_13*tmp.r33 + gamma_23*n_b*tmp.r22 - gamma_23*tmp.r33*(n_b + 1);
        diff->r12 = -Omega_mui*tmp.r13 - Omega_mur*tmp.i13 + Omega_oi*tmp.r11 - Omega_oi*tmp.r22 - Omega_pi*tmp.r23 - Omega_pr*tmp.i23 - deltap_o*tmp.i12 - 1.0/2.0*gamma_12*tmp.r12 - 1.0/2.0*gamma_23*n_b*tmp.r12 - 1.0/2.0*gamma_2d*tmp.r12;
        diff->i12 = -Omega_mui*tmp.i13 + Omega_mur*tmp.r13 + Omega_or*tmp.r11 - Omega_or*tmp.r22 + Omega_pi*tmp.i23 - Omega_pr*tmp.r23 + deltap_o*tmp.r12 - 1.0/2.0*gamma_12*tmp.i12 - 1.0/2.0*gamma_23*n_b*tmp.i12 - 1.0/2.0*gamma_2d*tmp.i12;
        diff->r13 = Omega_mui*tmp.r12 - Omega_mur*tmp.i12 - Omega_oi*tmp.r23 + Omega_or*tmp.i23 + Omega_pi*tmp.r11 - Omega_pi*tmp.r33 - 1.0/2.0*gamma_13*tmp.r13 - 1.0/2.0*gamma_23*tmp.r13*(n_b + 1) - 1.0/2.0*gamma_3d*tmp.r13 + tmp.i13*(-deltap_mu - deltap_o);
        diff->i13 = Omega_mui*tmp.i12 + Omega_mur*tmp.r12 - Omega_oi*tmp.i23 - Omega_or*tmp.r23 + Omega_pr*tmp.r11 - Omega_pr*tmp.r33 - 1.0/2.0*gamma_13*tmp.i13 - 1.0/2.0*gamma_23*tmp.i13*(n_b + 1) - 1.0/2.0*gamma_3d*tmp.i13 - tmp.r13*(-deltap_mu - deltap_o);
        diff->r23 = Omega_mui*tmp.r22 - Omega_mui*tmp.r33 + Omega_oi*tmp.r13 + Omega_or*tmp.i13 + Omega_pi*tmp.r12 + Omega_pr*tmp.i12 + deltap_o*tmp.i23 - 1.0/2.0*gamma_12*tmp.r23 - 1.0/2.0*gamma_13*tmp.r23 - 1.0/2.0*gamma_23*n_b*tmp.r23 - 1.0/2.0*gamma_23*tmp.r23*(n_b + 1) - 1.0/2.0*gamma_2d*tmp.r23 - 1.0/2.0*gamma_3d*tmp.r23 + tmp.i23*(-deltap_mu - deltap_o);
        diff->i23 = Omega_mur*tmp.r22 - Omega_mur*tmp.r33 + Omega_oi*tmp.i13 - Omega_or*tmp.r13 - Omega_pi*tmp.i12 + Omega_pr*tmp.r12 - deltap_o*tmp.r23 - 1.0/2.0*gamma_12*tmp.i23 - 1.0/2.0*gamma_13*tmp.i23 - 1.0/2.0*gamma_23*n_b*tmp.i23 - 1.0/2.0*gamma_23*tmp.i23*(n_b + 1) - 1.0/2.0*gamma_2d*tmp.i23 - 1.0/2.0*gamma_3d*tmp.i23 - tmp.r23*(-deltap_mu - deltap_o);
    }
}

__device__
void Omega(double *Omega_r, double *Omega_i,
           double g_r, double g_i,
           double alpha_r, double alpha_i)
{

    double abs_alpha_sqr = alpha_r*alpha_r + alpha_i*alpha_i;
    if (abs_alpha_sqr == 0.0)
    {
        *Omega_r = g_r;
        *Omega_i = g_i;
        return;
    }

    double alpha_rescale = sqrt(1.0 + 1.0/abs_alpha_sqr);
    alpha_r *= alpha_rescale;
    alpha_i *= alpha_rescale;

    *Omega_r = g_r*alpha_r - g_i*alpha_i;
    *Omega_i = g_r*alpha_i + g_i*alpha_r;
}

__global__
void EnsembleDerivativeAndSum(size_t nAtomSamples, SystemState *diff,
    const SystemState *state, const AtomSample *atomSamples, SystemType sys,
    double alpha_r, double alpha_i, double beta_r, double beta_i,
    double Omega_pr, double Omega_pi, double delta_mu, double delta_o,
    double n_b, double gamma_12, double gamma_13,
    double gamma_23, double gamma_2d, double gamma_3d)
{
    size_t threadId = blockIdx.x*blockDim.x + threadIdx.x;
    size_t threadCount = gridDim.x*blockDim.x;
    for (size_t k = threadId; k < nAtomSamples; k += threadCount)
    {
        // get sample-specific variables
        double g_or = atomSamples[k].g_or;
        double g_oi = atomSamples[k].g_oi;
        double g_mur = atomSamples[k].g_mur;
        double g_mui = atomSamples[k].g_mui;
        double delta_12 = atomSamples[k].delta_12;
        double delta_23 = atomSamples[k].delta_23;
        double w = atomSamples[k].weight;
        const DensityMatrix *rho = &(state->rho[k]);
        DensityMatrix *rhoDiff = &(diff->rho[k]);

        // contribution to ensemble terms
        double rho_or = ((sys == V_SYSTEM) ? rho->r12 : rho->r23);
        double rho_oi = ((sys == V_SYSTEM) ? -rho->i12 : -rho->i23);
        double rho_mur = ((sys == V_SYSTEM) ? rho->r23 : rho->r12);
        double rho_mui = ((sys == V_SYSTEM) ? -rho->i23 : -rho->i12);
        double S_alpha_r = w * (g_or*rho_or + g_oi*rho_oi);
        double S_alpha_i = w * (g_oi*rho_or - g_or*rho_oi);
        double S_beta_r = w * (g_mur*rho_mur + g_mui*rho_mui);
        double S_beta_i = w * (g_mui*rho_mur - g_mur*rho_mui);
        atomicAdd(&(diff->alpha_r), S_alpha_i);
        atomicAdd(&(diff->alpha_i), -S_alpha_r);
        atomicAdd(&(diff->beta_r), S_beta_i);
        atomicAdd(&(diff->beta_i), -S_beta_r);

        // drive Rabi frequencies
        double Omega_mur;
        double Omega_mui;
        double Omega_or;
        double Omega_oi;
        Omega(&Omega_mur, &Omega_mui, g_mur, g_mui, beta_r, beta_i);
        Omega(&Omega_or, &Omega_oi, g_or, g_oi, alpha_r, alpha_i);

        // inhomogeneous shift
        double deltap_mu = delta_mu + (sys == V_SYSTEM ? delta_23 : delta_12);
        double deltap_o = delta_o + (sys == V_SYSTEM ? delta_12 : delta_23);

        MasterDerivative(
            rhoDiff,
            sys,
            rho,
            Omega_mur,
            Omega_mui,
            Omega_or,
            Omega_oi,
            Omega_pr,
            Omega_pi,
            deltap_mu,
            deltap_o,
            n_b,
            gamma_12,
            gamma_13,
            gamma_23,
            gamma_2d,
            gamma_3d);
    }
}

void SystemDerivative(size_t nAtomSamples,
    SystemState *diff, const SystemState *state,
    const AtomSample *atomSamples, SystemType sys,
    double Omega_pr, double Omega_pi, double delta_mu, double delta_o,
    double n_b, double gamma_12, double gamma_13, double gamma_23,
    double gamma_2d, double gamma_3d, double gamma_o, double gamma_mu)
{
    double alpha_r;
    double alpha_i;
    double beta_r;
    double beta_i;
    cudaMemcpy(&alpha_r, &(state->alpha_r),
        sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&alpha_i, &(state->alpha_i),
        sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&beta_r, &(state->beta_r),
        sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&beta_i, &(state->beta_i),
        sizeof(double), cudaMemcpyDeviceToHost);
    
    double d_alpha_r = -alpha_r * gamma_o/2.0;
    double d_alpha_i = -alpha_i * gamma_o/2.0;
    double d_beta_r = -beta_r * gamma_mu/2.0;
    double d_beta_i = -beta_i * gamma_mu/2.0;
    cudaMemcpy(&(diff->alpha_r), &d_alpha_r,
        sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(&(diff->alpha_i), &d_alpha_i,
        sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(&(diff->beta_r), &d_beta_r,
        sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(&(diff->beta_i), &d_beta_i,
        sizeof(double), cudaMemcpyHostToDevice);
    
    EnsembleDerivativeAndSum<<<NUM_THREAD_BLOCKS,NUM_THREADS_IN_BLOCK>>>(
        nAtomSamples,
        diff,
        state,
        atomSamples,
        sys,
        alpha_r,
        alpha_i,
        beta_r,
        beta_i,
        Omega_pr,
        Omega_pi,
        delta_mu,
        delta_o,
        n_b,
        gamma_12,
        gamma_13,
        gamma_23,
        gamma_2d,
        gamma_3d);
    cudaDeviceSynchronize();
}

__global__
void GlobalDensityMatricesWeightedSum(size_t nAtomSamples,
    DensityMatrix *dst, const DensityMatrix *a, double aw,
    const DensityMatrix *b, double bw)
{
    size_t threadId = blockIdx.x*blockDim.x + threadIdx.x;
    size_t threadCount = gridDim.x*blockDim.x;
    for (size_t k = threadId; k < nAtomSamples; k += threadCount)
    {
        dst[k].r11 = aw*a[k].r11 + bw*b[k].r11;
        dst[k].r22 = aw*a[k].r22 + bw*b[k].r22;
        dst[k].r33 = aw*a[k].r33 + bw*b[k].r33;
        dst[k].r12 = aw*a[k].r12 + bw*b[k].r12;
        dst[k].i12 = aw*a[k].i12 + bw*b[k].i12;
        dst[k].r13 = aw*a[k].r13 + bw*b[k].r13;
        dst[k].i13 = aw*a[k].i13 + bw*b[k].i13;
        dst[k].r23 = aw*a[k].r23 + bw*b[k].r23;
        dst[k].i23 = aw*a[k].i23 + bw*b[k].i23;
    }
}

__global__
void GlobalCavityStateWeightedSum(size_t nAtomSamples,
    SystemState *dst, const SystemState *a, double aw,
    const SystemState *b, double bw)
{
    dst->alpha_r = aw*a->alpha_r + bw*b->alpha_r;
    dst->alpha_i = aw*a->alpha_i + bw*b->alpha_i;
    dst->beta_r = aw*a->beta_r + bw*b->beta_r;
    dst->beta_i = aw*a->beta_i + bw*b->beta_i;
}

void SystemStateWeightedSum(size_t nAtomSamples, SystemState *dst,
    const SystemState *a, double aw, const SystemState *b, double bw)
{
    GlobalCavityStateWeightedSum<<<1,1>>>(nAtomSamples, dst, a, aw, b, bw);
    cudaDeviceSynchronize();
    GlobalDensityMatricesWeightedSum<<<NUM_THREAD_BLOCKS,
        NUM_THREADS_IN_BLOCK>>>(nAtomSamples,
        dst->rho, a->rho, aw, b->rho, bw);
    cudaDeviceSynchronize();
}

void SystemRK4Step(size_t nAtomSamples, SystemState *__restrict__ state,
    const AtomSample *__restrict__ atomSamples,
    SystemState *__restrict__ tmp1, SystemState *__restrict__ tmp2,
    SystemType sys, double dt, double Omega_pr, double Omega_pi,
    double delta_mu, double delta_o, double n_b, double gamma_12,
    double gamma_13, double gamma_23, double gamma_2d,
    double gamma_3d, double gamma_o, double gamma_mu)
{
    SystemDerivative( // k1
        nAtomSamples,
        tmp1,
        state,
        atomSamples,
        sys,
        Omega_pr,
        Omega_pi,
        delta_mu,
        delta_o,
        n_b,
        gamma_12,
        gamma_13,
        gamma_23,
        gamma_2d,
        gamma_3d,
        gamma_o,
        gamma_mu);
    SystemStateWeightedSum(nAtomSamples, tmp2, state, 1.0, tmp1, dt/6.0);
    SystemStateWeightedSum(nAtomSamples, tmp1, state, 1.0, tmp1, dt/2.0);
    SystemDerivative( // k2
        nAtomSamples,
        tmp1,
        tmp1,
        atomSamples,
        sys,
        Omega_pr,
        Omega_pi,
        delta_mu,
        delta_o,
        n_b,
        gamma_12,
        gamma_13,
        gamma_23,
        gamma_2d,
        gamma_3d,
        gamma_o,
        gamma_mu);
    SystemStateWeightedSum(nAtomSamples, tmp2, tmp1, dt/3.0, tmp2, 1.0);
    SystemStateWeightedSum(nAtomSamples, tmp1, state, 1.0, tmp1, dt/2.0);
    SystemDerivative( // k3
        nAtomSamples,
        tmp1,
        tmp1,
        atomSamples,
        sys,
        Omega_pr,
        Omega_pi,
        delta_mu,
        delta_o,
        n_b,
        gamma_12,
        gamma_13,
        gamma_23,
        gamma_2d,
        gamma_3d,
        gamma_o,
        gamma_mu);
    SystemStateWeightedSum(nAtomSamples, tmp2, tmp1, dt/3.0, tmp2, 1.0);
    SystemStateWeightedSum(nAtomSamples, tmp1, state, 1.0, tmp1, dt);
    SystemDerivative( // k4
        nAtomSamples,
        tmp1,
        tmp1,
        atomSamples,
        sys,
        Omega_pr,
        Omega_pi,
        delta_mu,
        delta_o,
        n_b,
        gamma_12,
        gamma_13,
        gamma_23,
        gamma_2d,
        gamma_3d,
        gamma_o,
        gamma_mu);
    SystemStateWeightedSum(nAtomSamples, state, tmp1, dt/6.0, tmp2, 1.0);
}

double PlanckExcitation(double T, double omega)
{
    return 1.0 / expm1(HBAR*omega / (K_B*T));
}

double RandomUniform(void)
{
    return ((double) rand()) / ((double) RAND_MAX);
}

double RandomGaussian(void)
{
    double U1 = RandomUniform();
    double U2 = RandomUniform();

    // U1 == 0.0 results in an error when taking its log
    while (U1 == 0.0)
    {
        U1 = RandomUniform();
    }
    
    double R = sqrt(-2.0 * log(U1));
    double Theta = 2.0*PI * U2;
    return R * cos(Theta);
}

int main(void)
{
    // system constants
    const SystemType sys = LAMBDA_SYSTEM;
    const double omega_12 = 2.0*PI*5.186e9;
    const double d13 = 1.63e-32;
    const double d23 = 1.15e-32;
    const double tau_12 = 11.0;
    const double tau_3 = 0.011;
    const double gamma_2d = 1e6;
    const double gamma_3d = 1e6;
    const double sigma_o = 2.0*PI*419e6;
    const double sigma_mu = 2.0*PI*5e6;
    const double N = 1e16;
    const double gamma_oi = 2.0*PI*7.95e6;
    const double gamma_oc = 2.0*PI*1.7e6;
    const double gamma_mui = 2.0*PI*650e3;
    const double gamma_muc = 2.0*PI*1.5e6;
    const double g_or = 51.9;
    const double g_oi = 0.0;
    const double g_mur = 1.04;
    const double g_mui = 0.0;

    const double T = 4.6;
    const double n_b = PlanckExcitation(T, omega_12);
    const double tau_13 = tau_3 * d13*d13 / (d13*d13 + d23*d23);
    const double tau_23 = tau_3 * d23*d23 / (d13*d13 + d23*d23);
    const double gamma_12 = 1.0 / (tau_12*(n_b+1.0));
    const double gamma_13 = 1.0 / tau_13;
    const double gamma_23 = 1.0 / tau_23;

    const double sigma_12 = ((sys == V_SYSTEM) ? sigma_o : sigma_mu);
    const double sigma_23 = ((sys == V_SYSTEM) ? sigma_mu : sigma_o);

    // parameters
    size_t nAtomSamples = 1'000'000;
    const double weight = N / ((double) nAtomSamples);
    const double delta_mu = 2.0*sigma_mu;
    const double delta_o = 2.0*sigma_o;
    const double Omega_pr = 35000.0;
    const double Omega_pi = 0.0;
    const double alpha_0r = 1.0;
    const double alpha_0i = 0.0;
    const double beta_0r = 1.0;
    const double beta_0i = 0.0;

    // allocate arrays
    AtomSample *atomSamples;
    if (!(HostAllocateAtomSamples(&atomSamples, nAtomSamples)))
    {	
        printf("Host memory allocation failure\n");
        return -1;
    }

    AtomSample *deviceAtomSamples;
    if (!(DeviceAllocateAtomSamples(&deviceAtomSamples, nAtomSamples)))
    {
        free(atomSamples);
        printf("Device memory allocation failure\n");
        return -1;
    }

    SystemState *state;
    if (!(HostAllocateStateStruct(&state, nAtomSamples)))
    {
        free(atomSamples);
        cudaFree(deviceAtomSamples);
        printf("Host memory allocation failure\n");
        return -1;
    }

    SystemState *deviceState;
    if (!(DeviceAllocateStateStruct(&deviceState, nAtomSamples)))
    {
        free(atomSamples);
        cudaFree(deviceAtomSamples);
        free(state);
        printf("Device memory allocation failure\n");
        return -1;
    }

    SystemState *tmp1;
    if (!(DeviceAllocateStateStruct(&tmp1, nAtomSamples)))
    {
        free(atomSamples);
        cudaFree(deviceAtomSamples);
        free(state);
        cudaFree(deviceState);
        printf("Device memory allocation failure\n");
        return -1;
    }

    SystemState *tmp2;
    if (!(DeviceAllocateStateStruct(&tmp2, nAtomSamples)))
    {
        free(atomSamples);
        cudaFree(deviceAtomSamples);
        free(state);
        cudaFree(deviceState);
        cudaFree(tmp1);
        printf("Device memory allocation failure\n");
        return -1;
    }

    // populate arrays and copy to device memory
    state->alpha_r = alpha_0r;
    state->alpha_i = alpha_0i;
    state->beta_r = beta_0r;
    state->beta_i = beta_0i;

    for (size_t k = 0; k < nAtomSamples; k++)
    {
        atomSamples[k].g_or = g_or;
        atomSamples[k].g_oi = g_oi;
        atomSamples[k].g_mur = g_mur;
        atomSamples[k].g_mui = g_mui;
        atomSamples[k].weight = weight;
        atomSamples[k].delta_12 = sigma_12 * RandomGaussian();
        atomSamples[k].delta_23 = sigma_23 * RandomGaussian();

        state->rho[k] = GROUND_STATE_MATRIX;
    }

    CopyAtomSamples(deviceAtomSamples, atomSamples, nAtomSamples,
                    cudaMemcpyHostToDevice);
    CopyStateStruct(deviceState, state, nAtomSamples, cudaMemcpyHostToDevice);

    // run simulation and save results to binary files
    const char *dirname = ".";
    char atomSamplesFpath[256];
    sprintf(atomSamplesFpath, "%s/atom_samples", dirname);
    FILE *atomSamplesFp = fopen(atomSamplesFpath, "wb");
    fwrite(atomSamples, sizeof(AtomSample), nAtomSamples, atomSamplesFp);
    printf("Saved atom sample data as %s\n", atomSamplesFpath);
    fclose(atomSamplesFp);

    const double dt = 10e-12;
    size_t numSteps = 0;

    size_t numPrint = 0;
    while (true)
    {
        size_t toPrint;
        if (numPrint < 100)
        {
            toPrint = numPrint;
        }
        else
        {
            size_t n = (numPrint-100) / 900;
            toPrint = (numPrint-100) % 900 + 100;
            for (size_t i = 0; i < n; i++)
            {
                toPrint *= 10;
            }
        }

        while (numSteps < toPrint)
        {
            SystemRK4Step(
                nAtomSamples,
                deviceState,
                deviceAtomSamples,
                tmp1,
                tmp2,
                sys,
                dt,
                Omega_pr,
                Omega_pi,
                delta_mu,
                delta_o,
                n_b,
                gamma_12,
                gamma_13,
                gamma_23,
                gamma_2d,
                gamma_3d,
                gamma_oi+gamma_oc,
                gamma_mui+gamma_muc);
            numSteps++;
        }

        
        char fname[256];
        sprintf(fname, "state_step_%zd_dt_%zd_ps",
            numSteps, (size_t) round(dt*1e12));
        char fpath[256];
        sprintf(fpath, "%s/%s", dirname, fname);

        FILE *fp = fopen(fpath, "wb");
        CopyStateStruct(state, deviceState,
            nAtomSamples, cudaMemcpyDeviceToHost);
        fwrite(state, SizeofStateStruct(nAtomSamples), 1, fp);
        fclose(fp);

        printf("\nSaved step %zd as %s\n", numSteps, fpath);
        printf("alpha_r %f\n", state->alpha_r);
        printf("alpha_i %f\n", state->alpha_i);
        printf("beta_r %f\n", state->beta_r);
        printf("beta_i %f\n", state->beta_i);
        printf("rho[0].r11 %f\n", state->rho[0].r11);
        printf("rho[0].r22 %f\n", state->rho[0].r22);
        printf("rho[0].r33 %f\n", state->rho[0].r33);
        printf("rho[0].r12 %f\n", state->rho[0].r12);
        printf("rho[0].i12 %f\n", state->rho[0].i12);
        printf("rho[0].r13 %f\n", state->rho[0].r13);
        printf("rho[0].i13 %f\n", state->rho[0].i13);
        printf("rho[0].r23 %f\n", state->rho[0].r23);
        printf("rho[0].i23 %f\n", state->rho[0].i23);

        numPrint++;
    }

    free(atomSamples);
    cudaFree(deviceAtomSamples);
    free(state);
    cudaFree(deviceState);
    cudaFree(tmp1);
    cudaFree(tmp2);
    return 0;
}
