# This file will help creating ML Application as a package. 

from setuptools import find_packages, setup
from typing import List


HYPHEN_E_DOT='-e .'

def get_requirements(file_path: str) -> List[str]:
    '''
    get_requirements function will return the list of required libraries mentioned in file at file_path.
    '''

    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readline()
        requirements = [req.replace("\n", "") for req in requirements]

    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)

    return requirements


setup(
    name="StudentPerformancePredictor",
    version='0.0.1',
    author="Manan Rastogi",
    packages=find_packages(),
    # will search for __init__.py files in folders and make them available as package 

    install_requirements= get_requirements('requirements.txt'),
)


