import rclpy
from rclpy.node import Node
import torch
import numpy as np
from sensor_msgs.msg import Image, PointCloud2, PointField, CameraInfo
from cv_bridge import CvBridge

from std_msgs.msg import Header
import cv2
from tf2_ros import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
import yaml

from MonoDepth.depth_anything_v2.dpt import DepthAnythingV2

class DepthProcessorNode(Node):
    def __init__(self):
        super().__init__('depth_processor')
        self.declare_parameter('camera_info_path', '/home/rahgirrafi/ws_ESP32_CAM_MonoDepth/src/MonoDepth/config/camera_calibration.yaml')
        self._load_camera_info()
        self.declare_parameters(namespace='',
            parameters=[
                ('image_topic', 'camera/image'),
                ('depth_image_topic', 'camera/depth'),
                ('pointcloud_topic', 'camera/points'),
                ('fx', self.fx),
                ('fy', self.fy),
                ('cx', self.cx),
                ('cy', self.cy),
                ('depth_scale', 0.1),
                ('downsample_factor', 2),
                ('colormap', 11),
                ('encoder_type', 'vits')  # vits, vitb, vitl, vitg

            ])
        
        self.model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        
        self.device = self._get_device()
        self.depthmap = None
        self.bridge = CvBridge()

        
        self.subscription = self.create_subscription(
            Image,
            self.get_parameter('image_topic').value,
            self.image_callback,
            10
        )
        self.depth_image_pub = self.create_publisher(
            Image,
            self.get_parameter('depth_image_topic').value,
            10
        )
        self.pointcloud_pub = self.create_publisher(
            PointCloud2,
            self.get_parameter('pointcloud_topic').value,
            10
        )
        
        # Initialize AI model
        self._init_model()
        self.get_logger().info("Depth processor ready")
        self.tf_broadcaster = StaticTransformBroadcaster(self)
        self._publish_static_tf()

    def _load_camera_info(self):
        """Load camera calibration from YAML file"""
        try:
            path = self.get_parameter('camera_info_path').value
            with open(path, 'r') as file:
                calib_data = yaml.safe_load(file)
            
            self.camera_info = CameraInfo()
            self.camera_info.header.frame_id = "camera_frame"
            self.camera_info.height = calib_data['image_height']
            self.camera_info.width = calib_data['image_width']
            self.camera_info.k = calib_data['camera_matrix']['data']
            self.camera_info.d = calib_data['distortion_coefficients']['data']
            self.camera_info.r = calib_data['rectification_matrix']['data']
            self.camera_info.p = calib_data['projection_matrix']['data']
            self.camera_info.distortion_model = calib_data['distortion_model']
            
            camera_matrix = calib_data['camera_matrix']['data']
            self.fx, self.fy, self.cx, self.cy = camera_matrix[0], camera_matrix[4], camera_matrix[2], camera_matrix[5]
            
            self.get_logger().info(f"Loaded camera calibration from {path}")
            
        except Exception as e:
            self.get_logger().error(f"Failed to load camera info: {str(e)}")
            raise

    def _get_device(self):
        """Determine available compute device"""
        if torch.cuda.is_available():
            return 'cuda'
        if torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'

    def _publish_static_tf(self):
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = "map"  # Your global fixed frame
        transform.child_frame_id = "camera_frame"
        transform.transform.translation.z = 0.0  # Adjust based on your setup
        transform.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(transform)

    def _init_model(self):
        """Initialize Depth Anything V2 model"""
        try:
            encoder_type = self.get_parameter('encoder_type').value
            config = self.model_configs[encoder_type]
            
            self.model = DepthAnythingV2(**config)
            checkpoint_path = '/home/rahgirrafi/ws_ESP32_CAM_MonoDepth/src/MonoDepth/checkpoints/depth_anything_v2_vits.pth'
            
            # Load pretrained weights
            self.model.load_state_dict(
                torch.load(checkpoint_path, map_location='cpu')
            )
            
            self.model = self.model.to(self.device).eval()
            self.get_logger().info(f"Loaded Depth Anything V2 ({encoder_type}) on {self.device}")     
        except Exception as e:
            self.get_logger().error(f"Model initialization failed: {str(e)}")
            raise
        

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            depth_map = self._process_depth(frame)
            if depth_map is None:
                self.get_logger().error("Failed to generate depth map")
                return
                
            if depth_map.shape != frame.shape[:2]:
                self.get_logger().error(f"Depth map shape mismatch: {depth_map.shape} vs {frame.shape[:2]}")
                return
            self._publish_depth_image(depth_map, msg.header)
            pointcloud = self._create_pointcloud(depth_map, frame)
            self.pointcloud_pub.publish(pointcloud)
        except Exception as e:
            self.get_logger().error(f"Processing error: {str(e)}")
    
    def _publish_depth_image(self, depth_map, header):
        try:
            # Normalize and apply colormap
            depth_visual = cv2.normalize(
                depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_visual = cv2.applyColorMap(
                depth_visual, self.get_parameter('colormap').value)
            
            # Create and publish message
            depth_msg = self.bridge.cv2_to_imgmsg(depth_visual, "bgr8")
            depth_msg.header = header
            self.depth_image_pub.publish(depth_msg)

        except Exception as e:
            self.get_logger().error(f"Depth image publishing error: {str(e)}")
   
    def _process_depth(self, frame):
        """Process frame with Depth Anything V2"""
        try:
            # Original dimensions
            h, w = frame.shape[:2]
            patch_size = 14  # ViT patch size

            # Calculate padding needed
            pad_h = (patch_size - h % patch_size) % patch_size
            pad_w = (patch_size - w % patch_size) % patch_size

            # Pad image symmetrically
            padded_frame = cv2.copyMakeBorder(
                frame,
                top=pad_h//2,
                bottom=pad_h - pad_h//2,
                left=pad_w//2,
                right=pad_w - pad_w//2,
                borderType=cv2.BORDER_REFLECT
            )

            # Convert to tensor and normalize
            frame_tensor = torch.from_numpy(padded_frame).permute(2, 0, 1).float().to(self.device)
            frame_tensor = frame_tensor.unsqueeze(0) / 255.0

            # Perform inference
            with torch.no_grad():
                depth_tensor = self.model(frame_tensor)

            # Convert to numpy and crop back to original size
            depth_map = depth_tensor.squeeze().cpu().numpy()
            depth_map = depth_map[pad_h//2:pad_h//2 + h, pad_w//2:pad_w//2 + w]

            return depth_map

        except Exception as e:
            self.get_logger().error(f"Depth processing failed: {str(e)}")
            return None


    def _create_pointcloud(self, depth_map, color_frame):
        points = []
        colors = []
        downsample = self.get_parameter('downsample_factor').value
        
        # Get parameters once to avoid repeated calls
        depth_scale = self.get_parameter('depth_scale').value
        fx = self.get_parameter('fx').value
        fy = self.get_parameter('fy').value
        cx = self.get_parameter('cx').value
        cy = self.get_parameter('cy').value
        
        for v in range(0, depth_map.shape[0], downsample):
            for u in range(0, depth_map.shape[1], downsample):
                z = depth_map[v, u] * depth_scale
                if z <= 0:
                    continue
                
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                
                b, g, r = color_frame[v, u]
                rgb = (int(r) << 16) | (int(g) << 8) | int(b)
                
                points.append([x, y, z])
                colors.append(rgb)
        
        # Use the node's frame_id parameter instead
        return self._create_pc2_msg(points, colors, "camera_frame")

    def _create_pc2_msg(self, points, colors, frame_id):
        header = Header(stamp=self.get_clock().now().to_msg(), frame_id=frame_id)
        
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]
        
        data = np.column_stack((points, colors)).astype(np.float32).tobytes()
        
        return PointCloud2(
            header=header,
            height=1,
            width=len(points),
            is_dense=False,
            is_bigendian=False,
            fields=fields,
            point_step=16,
            row_step=16 * len(points),
            data=data
        )

def main(args=None):
    rclpy.init(args=args)
    node = DepthProcessorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()