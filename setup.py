from setuptools import setup

setup(name='assistive_bandits',
      version='0.1.0',
      install_requires=['gym==0.9.4', 'numpy', 'scipy', 'rllab', 
'matplotlib', 'sklearn', 'pyprind']
)


extras_require={
'tf': ['tensorflow>=1.2.0'],
'tf_gpu': ['tensorflow-gpu>=1.2.0'],
}

