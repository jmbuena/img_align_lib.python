#
from setuptools import setup
#
setup(
     name='img_align_lib',
     # Current version of libobjlocator
     version='1.0.0',
     description='Python routines for vision packages',
     author='Jose Miguel Buenaposada',
     author_email='errbuena@gmail.com',
     # Packages included in the distribution
     packages=[
              'img_align.cost_functions',
              'img_align.motion_models',
              'img_align.object_models',
              'img_align.optimizers',
              'img_align.test',
              'img_align.utils'
              ],
     zip_safe=False,
     # Dependent packages (distributions)
     install_requires=["numpy"],
     )
