from setuptools import setup, Extension, find_packages
import numpy as np

module = Extension('cfilt', 
        sources = ['cfilt/cfiltmodule.c', 
            'cfilt/filter.c',
            'cfilt/arm/arm_biquad_cascade_df2T_init_f32.c', 
            'cfilt/arm/arm_biquad_cascade_df2T_f32.c',
            'cfilt/arm/arm_biquad_cascade_df2T_init_f64.c', 
            'cfilt/arm/arm_biquad_cascade_df2T_f64.c',
            ],
        include_dirs = ['./cfilt', './cfilt/arm', np.get_include()], 
        extra_compile_args=['-std=c99', '-DARM_MATH_CM4', '-DTEST', '-Xlinker -dead_strip'])

setup(name= 'cfilt',
        version = '1.0.2',
        description = 'This is a package',
        packages = find_packages(),
        ext_modules = [module])
