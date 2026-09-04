from setuptools import setup

package_name = 'detection_pkg'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sunrise',
    maintainer_email='user@todo.todo',
    description='Defect detection node using Horizon BPU model',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detection_node = detection_pkg.detection_node:main',
            'detection_node1 = detection_pkg.detection_node1:main',
            'infrared_detection_node = detection_pkg.infrared_detection_node:main',
        ],
    },
)
