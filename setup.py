from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="cad-env",
    version="0.2.0",
    author="CAD Environment Team",
    author_email="",
    description="Окружение для обучения LLM и RL агентов работе с CAD",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=[],
    entry_points={"console_scripts": [
        "cad-env-data=cad_env.learning.cli:main",
        "cad-env-train=cad_env.learning.train:main",
    ]},
    extras_require={
        "data": ["numpy>=1.23", "matplotlib>=3.6", "pillow>=9"],
        "api": ["fastapi>=0.100", "uvicorn>=0.20", "httpx>=0.24"],
        "learning": ["trl==0.24.0", "transformers>=4.56.1,<5", "peft>=0.17,<0.19", "datasets>=4,<5", "torch>=2.2", "accelerate>=1.4"],
        "legacy": requirements,
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
)

