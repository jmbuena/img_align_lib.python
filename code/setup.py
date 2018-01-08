#
from setuptools import setup
#
setup(
     name='img_alig',
     # Current version of img_align
     version='0.1.0',
     description='Algorithms for image alignment with direct methods',
     author='Jose Miguel Buenaposada',
     author_email='josemiguel.buenaposada@urjc.es',
     # Packages included in the distribution
     packages=[
              'img_align',
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
