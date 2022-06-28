import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setuptools.setup(
   name="swifr",
   version="2.0.2",
   author="Lauren Alpert Sugden, Kaileigh Ahlquist, Joseph Paik, Sohini Ramachandran",
   author_email="lauren.v.sugden@gmail.com",
   license="GNU General Public License v3 (GPLv3)",
   description="SWIF(r) - SWeep Inference Framework(controlling for correlation) - package for detecting adaptation in population data",
   long_description=long_description,
   long_description_content_type="text/markdown",
   url="https://github.com/ramachandran-lab/SWIFr",
   packages=setuptools.find_packages(),
   include_package_data=True,
   install_requires=[
           'cycler==0.10.0',
           'future==0.17.1',
            'joblib==0.13.2',
            'kiwisolver==1.1.0',
           #'matplotlib==3.0.2',
            'matplotlib==2.1.2',
           #'numpy==1.17.0',
            'numpy==1.22.0',
            'pyparsing==2.4.2',
            'python-dateutil==2.8.0',
           #'scikit-learn==0.21.3',
            'scikit-learn==0.20.4',
           #'scipy==1.3.1',
            'scipy==1.2.3',
            'six==1.12.0',
            'sklearn==0.0'
],
   entry_points={
           'console_scripts': [
                   'swifr_train = swifr_pkg.SWIFr_train:main',
                   'swifr_test = swifr_pkg.SWIFr:main',
                   'calibration = swifr_pkg.calibration:main'
                   ]
           },
   classifiers=[
       "Programming Language :: Python :: 3",
       "Programming Language :: Python :: 2.7",
       "Development Status :: 4 - Beta",
       "Intended Audience :: Science/Research",
       "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
       "Natural Language :: English",
       "Operating System :: MacOS :: MacOS X",
       "Operating System :: POSIX :: Linux",

   ],

    
)


