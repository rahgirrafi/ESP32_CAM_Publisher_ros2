from setuptools import find_packages, setup
from glob import glob
import os
package_name = 'MonoDepth'

setup(
    name=package_name,
    version='0.0.0',
    packages=[
        package_name,
        f'{package_name}.depth_anything_v2',
    ],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'checkpoints'), glob('checkpoints/*.pth')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rahgirrafi',
    maintainer_email='rahgirrafi@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'depth_cloud_node = MonoDepth.DepthStreamNode:main',
            'camera_receiver = MonoDepth.camera_receiver:main',
            'depth_processor = MonoDepth.depth_processor:main',
        ],
    },
)
