from setuptools import setup, find_packages
import os

# Ensure build directories exist
os.makedirs(".build/egg-info", exist_ok=True)

setup(
    name="whisper-service",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        line.strip()
        for line in open("requirements.txt")
        if line.strip() and not line.startswith("#")
    ],
    egg_info={
        "egg_base": ".build"
    }
) 