from setuptools import find_packages, setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='f1_tire_predictor',
      version="0.1.0",
      description="F1 Lap Time Prediction Model for Tire Strategy Analysis",
      license="MIT",
      author="Diego Botelho",
      author_email="diego_nbotelho@hotmail.com",
      # url="https://github.com/diegonbotelho/f1-tire-prediction",
      install_requires=requirements,
      packages=find_packages(),
      test_suite="tests",
      include_package_data=True,
      zip_safe=False)
