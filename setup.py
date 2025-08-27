from setuptools import setup, find_packages

setup(
    name="personal_health_dashboard",   # ✅ use underscore or dash
    version="0.1.0",                     # ✅ valid version format
    author="Bharath",
    author_email="bharthrajtarumani07@gmail.com",
    description="Personal Health Dashboard for Disease Prediction",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "streamlit",
        "dill"
    ],
)
