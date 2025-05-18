from setuptools import setup

__version__ = '3.0.2-wm'

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='lcu-driver',
    version=__version__,
    author='André Matos de Sousa',
    author_email='andrematosdesousa@gmail.com',
    maintainer='WordlessMeteor',
    maintainer_email='WordlessMeteor@gmail.com',
    license='MIT',
    url='https://github.com/WordlessMeteor/lcu-driver/',
    long_description_content_type='text/markdown',
    long_description=long_description,
    packages=['lcu_driver', 'lcu_driver.events'],
    install_requires=[
        'aiohttp',
        'psutil',
        'pandas',
        'wcwidth'
    ],
    project_urls={
        'Source': 'https://github.com/sousa-andre/lcu-driver/'
    }
)

