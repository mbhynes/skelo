import setuptools

with open("README.md", "r") as fh:
  long_description = fh.read()

setuptools.setup(
  name="skelo",
  version="0.1.5",
  author="Michael B Hynes",
  author_email="mike.hynes.rhymes@gmail.com",
  description="A scikit-learn interface to the Elo and Glicko2 rating systems",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/mbhynes/skelo",
  packages=setuptools.find_packages(),
  classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Topic :: Scientific/Engineering :: Information Analysis",
  ],
  python_requires='>=3',
  install_requires=[
    "numpy",
    "scikit-learn",
    "pandas",
    "glicko2>=2",
  ],
)
