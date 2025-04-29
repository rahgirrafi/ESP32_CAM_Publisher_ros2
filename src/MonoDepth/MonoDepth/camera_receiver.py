import rclpy
from rclpy.node import Node
import socket
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from threading import Thread
import yaml

class CameraReceiverNode(Node):
    def __init__(self):
        super().__init__('camera_receiver')
        
        # Parameter declaration with descriptions
        self.declare_parameter('port', 8000)
        self.declare_parameter('image_topic', 'camera/image')
        self.declare_parameter('receive_timeout', 2.0)
        self.declare_parameter('camera_info_topic', 'camera/camera_info')
        self.declare_parameter('camera_info_path', '/home/rahgirrafi/ws_ESP32_CAM_MonoDepth/src/MonoDepth/config/camera_calibration.yaml')
        
        # Initialize components
        self.bridge = CvBridge()
        self._load_camera_info()
        self._init_publisher()
        self._init_network()
        self._log_startup_info()

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
            
            self.get_logger().info(f"Loaded camera calibration from {path}")
            
        except Exception as e:
            self.get_logger().error(f"Failed to load camera info: {str(e)}")
            raise
        

    def _init_publisher(self):
        """Initialize ROS publisher with QoS settings"""
        self.image_pub = self.create_publisher(
            Image,
            self.get_parameter('image_topic').value,
            10
        )

        self.camera_info_pub = self.create_publisher(
            CameraInfo,
            self.get_parameter('camera_info_topic').value,
            10
        )

        self.get_logger().info(f"Publishers created for image and camera info")

    def _init_network(self):
        """Configure network components and start receiver thread"""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.settimeout(self.get_parameter('receive_timeout').value)
        
        port = self.get_parameter('port').value
        try:
            self.sock.bind(('0.0.0.0', port))
            self.sock.listen(1)
            self.get_logger().info(f"TCP server listening on port {port}")
        except OSError as e:
            self.get_logger().error(f"Failed to bind to port {port}: {str(e)}")
            raise
        
        self.network_thread = Thread(target=self._receive_frames, daemon=True)
        self.network_thread.start()
        self.get_logger().debug("Network receiver thread started")
    
    def _publish_camera_info(self, stamp):
        """Publish camera info with synchronized timestamp"""
        self.camera_info.header.stamp = stamp
        self.camera_info_pub.publish(self.camera_info)
        self.get_logger().debug("Published camera info", throttle_duration_sec=1)

    def _log_startup_info(self):
        """Log initial configuration parameters"""
        self.get_logger().info("Camera receiver initialized with parameters:")
        self.get_logger().info(f"  - Port: {self.get_parameter('port').value}")
        self.get_logger().info(f"  - Image Topic: {self.get_parameter('image_topic').value}")
        self.get_logger().info(f"  - Timeout: {self.get_parameter('receive_timeout').value}s")

    def _receive_frames(self):
        """Main frame reception loop"""
        self.get_logger().info("Waiting for camera connection...")
        try:
            conn, addr = self.sock.accept()
            client_ip = addr[0]
            self.get_logger().info(f"Accepted connection from {client_ip}")
            
            with conn:
                self._handle_client_connection(conn, client_ip)
                
        except socket.timeout:
            self.get_logger().warning("Socket timeout while waiting for connection")
        except Exception as e:
            self.get_logger().error(f"Connection error: {str(e)}")
        finally:
            self.get_logger().info("Network receiver thread exiting")

    def _handle_client_connection(self, conn, client_ip):
        self.get_logger().info(f"Handling connection from {client_ip}")
        """Handle an active client connection"""
        while rclpy.ok():
            try:
                self.get_logger().debug(f"Waiting for frame header from {client_ip}...")
                header = conn.recv(4)
                if not header:
                    self.get_logger().info(f"Client {client_ip} disconnected")
                    break
                
                if len(header) != 4:
                    self.get_logger().warning(
                        f"Invalid header length {len(header)} bytes from {client_ip}")
                    continue
                self.get_logger().info(f"Received header from {client_ip}: {header}")
                frame_len = int.from_bytes(header, byteorder='little')
                self.get_logger().info(f"Receiving frame ({frame_len} bytes) from {client_ip}")
                
                data = self._receive_frame_data(conn, frame_len, client_ip)
                if data is not None:
                    self._process_and_publish_frame(data, client_ip)
                    
            except socket.timeout:
                self.get_logger().warning(f"Timeout waiting for data from {client_ip}")
                continue
            except ConnectionResetError:
                self.get_logger().warning(f"Connection reset by {client_ip}")
                break

    def _receive_frame_data(self, conn, expected_length, client_ip):
        """Receive complete frame data"""
        data = bytearray()
        while len(data) < expected_length:
            try:
                packet = conn.recv(expected_length - len(data))
                if not packet:
                    self.get_logger().info(f"Client {client_ip} closed connection during transfer")
                    return None
                data.extend(packet)
            except socket.timeout:
                self.get_logger().error(
                    f"Timeout receiving data from {client_ip} "
                    f"(received {len(data)}/{expected_length} bytes)")
                return None
        
        self.get_logger().debug(f"Received complete frame from {client_ip}")
        return data

    def _process_and_publish_frame(self, data, client_ip):
        """Process image data and publish ROS message"""
        try:
            frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                self.get_logger().error("Failed to decode image frame")
                return
            
            # Update dynamic camera info parameters
            stamp = self.get_clock().now().to_msg()
            self.camera_info.height = frame.shape[0]
            self.camera_info.width = frame.shape[1]

            # Create and publish image message
            msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "camera_frame"
            
            self.image_pub.publish(msg)
            self._publish_camera_info(stamp)

            self.get_logger().debug(f"Published frame from {client_ip} "
                                  f"(resolution: {frame.shape[1]}x{frame.shape[0]})")
            
        except Exception as e:
            self.get_logger().error(f"Frame processing error: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = CameraReceiverNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutdown signal received")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        node.get_logger().info("Camera receiver node shutdown complete")

if __name__ == '__main__':
    main()