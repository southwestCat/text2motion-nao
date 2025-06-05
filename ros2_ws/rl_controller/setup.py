from setuptools import find_packages, setup
import os

package_name = 'rl_controller'

data_files = []
for root, dirs, files in os.walk(os.path.join(package_name, 'data')):
    for file in files:
        if file.endswith('.npy'):
            data_files.append(os.path.join(root, file))
policy_file = os.path.join(package_name, 'policy', 'policy.onnx')
config_file = os.path.join(package_name, 'config', 'config.yaml')

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/data', data_files),
        ('share/' + package_name + '/policy', [policy_file]),
        ('share/' + package_name + '/config', [config_file]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Author',
    maintainer_email='email',
    description='NAO RL controller',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rl_controller_node = rl_controller.rl_controller_node:main',
        ],
    },
)
