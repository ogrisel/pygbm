import os
import re
import shutil
from setuptools import setup, find_packages
from distutils.command.clean import clean as Clean

packages = find_packages(exclude=['tests', 'benchmark', 'examples'])


# Function to parse __version__ in `loky`
def find_version():
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, 'pygbm', '__init__.py'), 'r') as fp:
        version_file = fp.read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


# Custom clean command to remove build artifacts
class CleanCommand(Clean):
    description = "Remove build artifacts from the source tree"

    def run(self):
        Clean.run(self)
        # Remove c files if we are not within a sdist package
        cwd = os.path.abspath(os.path.dirname(__file__))
        remove_c_files = not os.path.exists(os.path.join(cwd, 'PKG-INFO'))
        if remove_c_files:
            print('Will remove generated .c files')
        if os.path.exists('build'):
            shutil.rmtree('build')
        for dirpath, dirnames, filenames in os.walk('.'):
            for filename in filenames:
                if any(filename.endswith(suffix) for suffix in
                       (".so", ".pyd", ".dll", ".pyc")):
                    os.unlink(os.path.join(dirpath, filename))
                    continue
                extension = os.path.splitext(filename)[1]
                if remove_c_files and extension in ['.c', '.cpp']:
                    pyx_file = str.replace(filename, extension, '.pyx')
                    if os.path.exists(os.path.join(dirpath, pyx_file)):
                        os.unlink(os.path.join(dirpath, filename))
            for dirname in dirnames:
                if dirname == '__pycache__':
                    shutil.rmtree(os.path.join(dirpath, dirname))


setup(
    name='pygbm',
    version=find_version(),
    description=('Experimental, numba-based Gradient Boosting Machines'),
    long_description=open('README.md', 'rb').read().decode('utf-8'),
    long_description_content_type='text/markdown',
    url='https://github.com/ogrisel/pygbm/',
    author='Olivier Grisel',
    author_email='olivier.grisel@ensta.org',
    packages=packages,
    zip_safe=False,
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
    ],
    cmdclass={'clean': CleanCommand},
    platforms='any',
    install_requires=['numpy', 'scipy', 'scikit-learn', 'numba'],
    python_requires='>=3.6',
    tests_require=['pytest'],
)
