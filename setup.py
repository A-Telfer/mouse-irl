from setuptools import setup, find_namespace_packages

setup(
    name="mouse_irl",
    version="0.0.2",
    install_requires=["gymnasium==0.27.1", "pygame==2.1.2"],
    packages=find_namespace_packages(where="src"),
    package_dir = {'': 'src'},
    package_data = {'mouse_irl.data': ['**/*.json']},
    include_package_data=True,
    exclude_package_data={"mypkg": [".gitattributes"]}
)