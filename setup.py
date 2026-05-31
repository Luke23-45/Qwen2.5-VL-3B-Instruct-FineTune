from setuptools import find_packages, setup


packages = sorted(
    set(find_packages("src") + find_packages(where=".", include=["inference*"]))
)

setup(
    packages=packages,
    package_dir={
        "dataset_pipeline": "src/dataset_pipeline",
        "vlm_training": "src/vlm_training",
        "inference": "inference",
    },
    include_package_data=True,
)
