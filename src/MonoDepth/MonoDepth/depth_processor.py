import rclpy
from rclpy.node import Node
import torch
import numpy as np
from sensor_msgs.msg import Image, PointCloud2, PointField
from cv_bridge import CvBridge

from std_msgs.msg import Header
import cv2
from tf2_ros import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped


class DepthProcessorNode(Node):
    def __init__(self):
        super().__init__('depth_processor')
        self.declare_parameters(namespace='',
            parameters=[
                ('image_topic', 'camera/image'),
                ('depth_image_topic', 'camera/depth'),
                ('pointcloud_topic', 'camera/points'),
                ('fx', 320.0),
                ('fy', 320.0),
                ('cx', 160.0),
                ('cy', 120.0),
                ('depth_scale', 0.1),
                ('downsample_factor', 2),
                ('colormap', 11)
            ])
        
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

    def _publish_static_tf(self):
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = "map"  # Your global fixed frame
        transform.child_frame_id = "camera_frame"
        transform.transform.translation.z = 0.0  # Adjust based on your setup
        transform.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(transform)

    def _init_model(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(self.device)
        self.model.eval()
        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            depth_map = self._process_depth(frame)
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
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img).to(self.device)
        
        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        return prediction.cpu().numpy()

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