from setuptools import setup, find_packages

setup(
    name="rag-ninia",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.1.0",
        "langchain-openai>=0.0.5",
        "openai>=1.0.0",
        "pandas>=2.0.0",
        "python-dotenv>=1.0.0",
        "streamlit>=1.0.0",
        "watchdog>=3.0.0",
        "unidecode>=1.0.0"
    ],
    python_requires=">=3.8",
) 