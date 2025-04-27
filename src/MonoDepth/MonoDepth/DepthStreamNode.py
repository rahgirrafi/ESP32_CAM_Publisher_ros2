
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import torch
import socket
import struct
from threading import Thread
from sensor_msgs.msg import Image, PointCloud2, PointField
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge
from std_msgs.msg import Header


# ROS2 PointCloud2 configuration
POINTCLOUD_DOWNSAMPLE = 2  # Process every nth pixel
FIELDS = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
]

class DepthCloudNode(Node):
    def __init__(self):
        super().__init__('depth_stream_node')
        
        # ROS2 Publishers
        self.image_pub = self.create_publisher(Image, 'esp32_cam/image_raw', 10)
        self.pointcloud_pub = self.create_publisher(PointCloud2, 'esp32_cam/points', 10)
        
        # Camera parameters (calibrate for your ESP32-CAM!)
        self.fx = 320.0  # Focal length x
        self.fy = 320.0  # Focal length y
        self.cx = 160.0  # Principal point x (QVGA width/2)
        self.cy = 120.0  # Principal point y (QVGA height/2)
        self.depth_scale = 0.1  # Scale factor for depth values
        
        # Network configuration
        self.setup_network()
        
        # AI model setup
        self.setup_ai_model()
        
        # Create ROS2 timer for processing
        self.timer = self.create_timer(0.1, self.process_data)

    def setup_network(self):
        self.bridge = CvBridge()
        self.current_frame = None
        self.current_depth = None
        
        # TCP Server setup
        #log
        self.get_logger().info("Setting up TCP server...")
        self.get_logger().info("Waiting for ESP32-CAM connection")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('0.0.0.0', 8000))
        #log
        self.get_logger().info("TCP server setup complete")
        
        self.sock.listen(1)
        
        # Start network thread
        self.network_thread = Thread(target=self.receive_frames)
        self.network_thread.start()

    def setup_ai_model(self):
        self.get_logger().info("Loading MiDaS model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_type = "MiDaS_small"
        self.midas = torch.hub.load("intel-isl/MiDaS", model_type).to(device)
        self.midas.eval()
        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
        self.get_logger().info("MiDaS model loaded successfully")


    def receive_frames(self):
        self.get_logger().info("Waiting for connection...")
        try:
            conn, addr = self.sock.accept()
            self.get_logger().info(f"Connected to {addr}")
            
            try:
                while rclpy.ok():   
                    # Read frame length header
                    self.get_logger().info("Waiting for frame header...")
                    header = conn.recv(4)
                    if len(header) != 4:
                        break
                        
                    frame_len = struct.unpack('>I', header)[0]
                    self.get_logger().info(f"Frame length: {frame_len}")
                    
                    # Read frame data
                    data = b''
                    while len(data) < frame_len:
                        self.get_logger().info(f"Receiving {frame_len - len(data)} bytes...")
                        packet = conn.recv(frame_len - len(data))
                        if not packet:
                            break
                        data += packet
                    
                    # Process frame
                    if len(data) == frame_len:
                        self.get_logger().info(f"Received frame of size: {len(data)}")
                        image = np.frombuffer(data, dtype=np.uint8)
                        self.current_frame = cv2.imdecode(image, cv2.IMREAD_COLOR)
                        self.process_depth()
            finally:
                conn.close()
                self.get_logger().info("Connection closed")
        except Exception as e:
            self.get_logger().error(f"Connection error: {str(e)}")

    def process_depth(self):
        self.get_logger().info("Processing depth...")
        if self.current_frame is None:
            return

        try:
            self
            # Convert to RGB and process
            img = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            input_batch = self.transform(img).to(next(self.midas.parameters()).device)
            
            with torch.no_grad():
                prediction = self.midas(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            
            self.current_depth = prediction.cpu().numpy()
        except Exception as e:
            self.get_logger().error(f"Depth processing failed: {str(e)}")

    def depth_to_pointcloud(self, depth_map, color_image):
        points = []
        colors = []
        
        height, width = depth_map.shape
        for v in range(0, height, POINTCLOUD_DOWNSAMPLE):
            for u in range(0, width, POINTCLOUD_DOWNSAMPLE):
                z = depth_map[v, u] * self.depth_scale
                if z <= 0:
                    continue
                
                # Convert to 3D coordinates
                x = (u - self.cx) * z / self.fx
                y = (v - self.cy) * z / self.fy
                
                # Get color (swap RGB -> BGR for ROS)
                b, g, r = color_image[v, u]
                rgb = (int(r) << 16) | (int(g) << 8) | int(b)
                
                points.append([x, y, z])
                colors.append(rgb)
        
        return np.array(points), np.array(colors)

    def create_pointcloud2(self, points, colors, frame_id="camera_frame"):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = frame_id

        dtype = np.float32
        itemsize = np.dtype(dtype).itemsize
        
        data = np.column_stack((points, colors)).astype(np.float32).tobytes()
        
        return PointCloud2(
            header=header,
            height=1,
            width=len(points),
            is_dense=False,
            is_bigendian=False,
            fields=FIELDS,
            point_step=16,  # 4 bytes per field * 4 fields
            row_step=16 * len(points),
            data=data
        )

    def process_data(self):
        if self.current_frame is None or self.current_depth is None:
            return

        try:
            # Publish original image
            img_msg = self.bridge.cv2_to_imgmsg(self.current_frame, "bgr8")
            img_msg.header.stamp = self.get_clock().now().to_msg()
            img_msg.header.frame_id = "camera_frame"
            self.image_pub.publish(img_msg)
            
            # Generate point cloud
            points, colors = self.depth_to_pointcloud(self.current_depth, self.current_frame)
            
            # Create and publish PointCloud2
            cloud_msg = self.create_pointcloud2(points, colors)
            self.pointcloud_pub.publish(cloud_msg)
            
        except Exception as e:
            self.get_logger().error(f"Processing error: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = DepthCloudNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()