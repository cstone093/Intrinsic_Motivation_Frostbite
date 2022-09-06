from setuptools import setup

setup(name='project',
      version='0.3',
      description='MSc RL project for a model-based intrinsic agent',
      url='https://github.com/cstone093/Model-Based-Intrinsic-Agent-Solaris',
      author='Charlotte Stone',
      packages=['project','project.agents','project.data_structures','project.environments','project.hyperparameters','project.models','project.policies'],
      zip_safe=False),
      