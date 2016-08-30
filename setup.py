from distutils.core import setup, Extension
import numpy as np

module = Extension('cfilt', 
        sources = ['cfiltmodule.c', 
            'filter.c',
            'arm/arm_biquad_cascade_df2T_init_f32.c', 
            'arm/arm_biquad_cascade_df2T_f32.c',
            'arm/arm_biquad_cascade_df2T_init_f64.c', 
            'arm/arm_biquad_cascade_df2T_f64.c',
            ],
        library_dirs=['./build/'],
        include_dirs = ['.', './arm', np.get_include()], 
        extra_compile_args=['-std=c99', '-DARM_MATH_CM4', '-DTEST', '-Xlinker -dead_strip'])

setup(name= 'CFILT',
        version = '1.0',
        description = 'This is a package',
        ext_modules = [module])
