from setuptools import setup

package_name = 'motor_control_pkg'

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
    description='Motor control node using GPIO',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'motor_control_node = motor_control_pkg.motor_control_node:main',
            'web_control_node = motor_control_pkg.web_control_node:main',
        ],
    },
)
