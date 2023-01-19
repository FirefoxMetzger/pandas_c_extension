from setuptools.extension import Extension
import numpy as np


custom_extension = Extension(
    "local._lib.transforms",
    sources=["local/_lib/transforms.c"],
    include_dirs=[
        np.get_include(),
    ],
    # library_dirs = [
    #     np.get_lib
    # ]
    define_macros=[("PY_SSIZE_T_CLEAN",)],
)


def build(setup_kwargs):
    setup_kwargs.update(
        {
            "ext_modules": [custom_extension],
        }
    )
