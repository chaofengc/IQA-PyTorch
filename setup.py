#!/usr/bin/env python
r"""Setup file for python package
Taken from: https://github.com/xinntao/BasicSR/blob/master/setup.py
"""
from setuptools import find_packages, setup
import os
import subprocess
import time

version_file = 'pyiqa/version.py'


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


def get_git_hash():

    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH', 'HOME']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        sha = out.strip().decode('ascii')
    except OSError:
        sha = 'unknown'

    return sha


def get_hash():
    if os.path.exists('.git'):
        sha = get_git_hash()[:7]
    else:
        sha = 'unknown'

    return sha


def write_version_py():
    content = """# GENERATED VERSION FILE
# TIME: {}
__version__ = '{}'
__gitsha__ = '{}'
version_info = ({})
"""
    sha = get_hash()
    with open('VERSION', 'r') as f:
        SHORT_VERSION = f.read().strip()
    VERSION_INFO = ', '.join([x if x.isdigit() else f'"{x}"' for x in SHORT_VERSION.split('.')])

    version_file_str = content.format(time.asctime(), SHORT_VERSION, sha, VERSION_INFO)
    with open(version_file, 'w') as f:
        f.write(version_file_str)


def get_version():
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


def get_requirements(filename='requirements.txt'):
    here = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(here, filename), 'r') as f:
        requires = [line.replace('\n', '') for line in f.readlines()]
    return requires


if __name__ == '__main__':
    write_version_py()
    setup(
        name='pyiqa',
        version=get_version(),
        description='PyTorch Toolbox for Image Quality Assessment',
        long_description=readme(),
        long_description_content_type='text/markdown',
        author='Chaofeng Chen',
        author_email='chaofenghust@gmail.com',
        keywords='image quality assessment, pytorch',
        url='https://github.com/chaofengc/IQA-PyTorch',
        include_package_data=True,
        packages=find_packages(
            exclude=('options', 'datasets', 'experiments', 'results', 'tb_logger', 'wandb', 'tests', 'ResultsCalibra',
                     'scripts')),
        classifiers=[
            # How mature is this project? Common values are
            #   3 - Alpha
            #   4 - Beta
            #   5 - Production/Stable
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Image Processing',
        ],
        install_requires=get_requirements(),
        python_requires='>=3.6',
    )
