from setuptools import find_packages
from setuptools import setup

with open('requirements.txt') as f:
    content = f.readlines()
requirements = [x.strip() for x in content if 'git+' not in x]

setup(name='draw-it-model',
      version="1.0",
      description="ML application to classify images based on hand-drawn sketches",
      packages=find_packages(),
      install_requires=requirements,
      include_package_data=True,
      zip_safe=False)
