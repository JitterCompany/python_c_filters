#include "filter.h"

void filter_init_64(Filter64 *f) 
{
    arm_biquad_cascade_df2T_init_f64(
            &f->filter_instance,
            f->num_stages,
            f->coeffs,
            f->state
        );
}

void filter_init_32(Filter32 *f) 
{
    arm_biquad_cascade_df2T_init_f32(
            &f->filter_instance,
            f->num_stages,
            f->coeffs,
            f->state
        );
}

void filter_apply_64(Filter64* f, double *src, double *dst, unsigned int n)
{
    if (dst == NULL) {
        dst = src;
    }
    arm_biquad_cascade_df2T_f64 (
        &(f->filter_instance),
        src,
        dst,
        n
    );
    
}

void filter_apply_32(Filter32* f, float *src, float *dst, unsigned int n)
{
    if (dst == NULL) {
        dst = src;
    }
    memcpy(dst, src, n*sizeof(float));
    arm_biquad_cascade_df2T_f32 (
        &(f->filter_instance),
        src,
        dst,
        n
    );
}
