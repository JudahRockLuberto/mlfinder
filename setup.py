# taken from http://python-packaging.readthedocs.io/en/latest/everything.html and modified a little

from setuptools import setup

# random values
__version__ = '0.1.0'

# this part taken from https://github.com/dr-guangtou/riker
with open('requirements.txt') as infd:
    INSTALL_REQUIRES = [x.strip('\n') for x in infd.readlines()]

# code taken from above

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='mlfinder',
      version=__version__,
      description='Find possible microlensing events.',
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Topic :: Text Processing :: Linguistic',
      ],
      keywords='astronomy',
      url='https://github.com/JudahRockLuberto/mlfinder',
      author='Judah Luberto',
      author_email='jluberto@ucsc.edu',
      license='MIT',
      packages=find_packages()
      ]
      },
      install_requires=INSTALL_REQUIRES,
      include_package_data=True,
      zip_safe=False,
      python_requires='>=3.6',
