from setuptools import setup,find_packages

with open("Requirements.txt") as f:
    Requirements=f.read().splitlines()

    setup(
        name="Personal Health Dashboard for Disease Prediction",
        version="0.1",
        author="Bharath",
        packages=find_packages(),
        install_requires=Requirements,
    )