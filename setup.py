from setuptools import setup

setup(
    name='hgame-averages',
    version='0.1',
    packages=['hgame.averages'],
    include_package_data=True,
    python_requires=">=3.7",
    install_requires=[
        'damm', 'xlrd', 'numpy', 'pandas', 'Click'
    ],
    entry_points="""
        [console_scripts]
        hgame-averages=hgame.averages.main:cli
    """
)
