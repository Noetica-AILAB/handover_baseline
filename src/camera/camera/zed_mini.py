import cv2
import rclpy
import pyzed.sl as sl
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo

# 设置相机参数
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
CAMERA_FPS = 30

class ZedMini(Node):
    def __init__(self):
        super().__init__("zed_mini")
        # 创建相机RGB图像的发布者
        self.left_pub = self.create_publisher(Image, "zed_mini/cam_left", 1)
        self.right_pub = self.create_publisher(Image, "zed_mini/cam_right", 1)
        self.right_info_pub = self.create_publisher(CameraInfo, "/zed_mini/right/camera_info", 1)
        self.bridge = CvBridge()
        # 创建相机并配置参数
        self.zed = sl.Camera()
        self.params = sl.InitParameters(
            camera_resolution = sl.RESOLUTION.HD720,
            camera_fps = CAMERA_FPS,
            depth_mode = sl.DEPTH_MODE.NONE,
            coordinate_units=sl.UNIT.METER,
        )
        # 打开相机并获取相机内参
        self.zed.open(self.params)
        self.calib = self.zed.get_camera_information().camera_configuration.calibration_parameters

    # 发布RGB图像（numpy）
    def image_publisher(self):
        left_image = sl.Mat()
        right_image = sl.Mat()

        self.zed.grab()
        self.zed.retrieve_image(left_image, sl.VIEW.LEFT)
        self.zed.retrieve_image(right_image, sl.VIEW.RIGHT)

        left_rgb = cv2.cvtColor(left_image.get_data(), cv2.COLOR_BGRA2RGB)
        right_rgb = cv2.cvtColor(right_image.get_data(), cv2.COLOR_BGRA2RGB)

        output_size = (IMAGE_WIDTH, IMAGE_HEIGHT)
        left_rgb = cv2.resize(left_rgb, output_size, interpolation=cv2.INTER_LINEAR)
        right_rgb = cv2.resize(right_rgb, output_size, interpolation=cv2.INTER_LINEAR)

        left_msg = self.bridge.cv2_to_imgmsg(left_rgb, encoding="rgb8")
        right_msg = self.bridge.cv2_to_imgmsg(right_rgb, encoding="rgb8")

        self.left_pub.publish(left_msg)
        self.right_pub.publish(right_msg)

    # 发布相机参数
    def camerainfo_publisher(self, stamp):
        config = self.zed.get_camera_information().camera_configuration
        calib = config.calibration_parameters
        right = calib.right_cam
        # 按发布尺寸缩放焦距，并按 OpenCV 的像素中心约定变换主点
        sx = IMAGE_WIDTH / config.resolution.width
        sy = IMAGE_HEIGHT / config.resolution.height
        fx = float(right.fx) * sx
        fy = float(right.fy) * sy
        cx = (float(right.cx) + 0.5) * sx - 0.5
        cy = (float(right.cy) + 0.5) * sy - 0.5
        baseline_m = float(calib.get_camera_baseline())
        
        msg = CameraInfo()
        msg.header.stamp = stamp
        msg.header.frame_id = "zed_mini_right_optical_frame"
        msg.width = IMAGE_WIDTH
        msg.height = IMAGE_HEIGHT

        # 描述已校正的图像：无畸变，校正矩阵为单位矩阵
        msg.distortion_model = "plumb_bob"
        msg.d = [0.0] * 5

        msg.k = [
            fx,  0.0, cx,
            0.0, fy,  cy,
            0.0, 0.0, 1.0,
        ]

        msg.r = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]

        # 右目投影矩阵：P[0] = fx，P[3] = -fx × B
        msg.p = [
            fx,  0.0, cx,  -fx * baseline_m,
            0.0, fy,  cy,  0.0,
            0.0, 0.0, 1.0, 0.0,
        ]

        self.right_info_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ZedMini()

    while rclpy.ok():
        node.image_publisher()
        stamp = node.get_clock().now().to_msg()
        node.camerainfo_publisher(stamp)

    node.zed.close()
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
