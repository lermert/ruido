from setuptools import setup, find_packages

setup(
    name = 'ruido',
    version = '0.0.0a0',
    description = 'Scripts for extracting measurements from ambient noise auto- and cross-correlation data',
    #long_description =
    # url = 
    author = 'L. Ermert, M. Denolle',
    author_email  = 'lermert@uw.edu',
    # license
    classifiers = [
        'Development Status :: 3 - Alpha',
        'Topic :: Seismology',
        'Programming Language :: Python :: 3',
    ],
    keywords = "Ambient seismic noise seismology Earth Sciences"
    packages = find_packages(),
    package_data={},
    install_requires = [
        "numpy",
        "scipy",
        "pandas",
        "h5py",
        "pytest",
        "mpi4py",
        "kneed"],
    entry_points = {
    },
)

