from setuptools import setup, find_packages

setup(
    name='NeuralFlow',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        # Add your project's dependencies here
        'numpy',
        'pandas',
        'tensorflow',
    ],
    # author='Your Name',
    # author_email='your.email@example.com',
    # description='A project for neural network flow analysis',
    # url='https://github.com/yourusername/NeuralFlow',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)