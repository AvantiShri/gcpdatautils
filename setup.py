from setuptools import setup

config = {
    'include_package_data': True,
    'description': 'Utilities for parsing GCP (Global Consciousness Project) Data',
    'download_url': 'https://github.com/AvantiShri/gcpdatautils/',
    'version': '0.1.1.1',
    'packages': ['gcpdatautils', 'gcpdatautils.resources'],
    'package_data': {'gcpdatautils.resources': ['rotteneggs.txt']},
    'setup_requires': [],
    'install_requires': ['bs4', 'h5py', 'numpy', 'scipy'],
    'dependency_links': [],
    'scripts': [],
    'name': 'gcpdatautils'
}

if __name__== '__main__':
    setup(**config)

