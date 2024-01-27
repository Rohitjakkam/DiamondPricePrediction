from setuptools import find_packages,setup
from typing import List

HYPEN_E_dot = '-e .'

def get_requirements(file_path:str)->List[str]:
    requiremnts=[]
    with open(file_path) as file_obj:
        requiremnts = file_obj.readlines()
        requiremnts=[req.replace("\n","") for req in requiremnts]

        if HYPEN_E_dot in requiremnts:
            requiremnts.remove(HYPEN_E_dot)
        return requiremnts


setup(
    name='DiamondPricePrediction',
    version='0.0.1',
    author='Rohitjakkam',
    author_email="rohitjakkakm@gmail.com",
    install_requires=get_requirements('requirements.txt'),
    packages=find_packages()
    )