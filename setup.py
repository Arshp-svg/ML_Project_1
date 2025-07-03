from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = '-e .'
def get_requirements(file_path:str) -> List[str]:
    """
    This function will return a list of requirements
    """
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.replace("\n","") for req in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
        
    return requirements

setup(
    name='MLproject',
    version='0.1',
    author='Arsh',
    author_email='arshpatel213@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    description='A simple ML project',

)