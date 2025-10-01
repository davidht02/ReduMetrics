from setuptools import setup, find_packages

setup(
    name="redu_metrics",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        # Aquí puedes poner tus dependencias si tienes, por ejemplo:
        # "numpy", "pandas", "scikit-learn"
    ],
    author="Tu Nombre",
    description="Librería para métricas de reducción de dimensionalidad",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)