#ifndef FILTER_H
#define FILTER_H

#include "arm_math.h"

#define MAX_FILTER_ORDER 16
#define COEFFS_PER_STAGE 5
#define MAX_NUM_COEFFS (MAX_FILTER_ORDER*COEFFS_PER_STAGE)

typedef struct {
    unsigned int num_stages;
    double coeffs[MAX_NUM_COEFFS];
    double state[2*MAX_FILTER_ORDER];
    arm_biquad_cascade_df2T_instance_f64 filter_instance;
} Filter64;


typedef struct {
    unsigned int num_stages;
    float coeffs[MAX_NUM_COEFFS];
    float state[2*MAX_FILTER_ORDER];
    arm_biquad_cascade_df2T_instance_f32 filter_instance;
} Filter32;

void filter_init_32(Filter32 *f);
void filter_init_64(Filter64 *f);

void filter_apply_64(Filter64* f, double *src, double *dst, unsigned int n);
void filter_apply_32(Filter32* f, float *src, float *dst, unsigned int n);
#endif
