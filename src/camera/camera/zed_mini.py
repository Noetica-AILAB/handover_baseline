import cv2
import rclpy
import pyzed.sl as sl
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

# 设置相机参数
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
CAMERA_FPS = 60

class ZedMini(Node):
    def __init__(self):
        super().__init__("zed_mini")
        # 创建相机RGB图像的发布者
        self.left_pub = self.create_publisher(Image, "zed_mini/cam_left", 1)
        self.right_pub = self.create_publisher(Image, "zed_mini/cam_right", 1)
        self.bridge = CvBridge()
        # 创建相机并配置参数
        self.zed = sl.Camera()
        self.params = sl.InitParameters(
            camera_resolution = sl.RESOLUTION.HD720,
            camera_fps = CAMERA_FPS,
            depth_mode = sl.DEPTH_MODE.NONE,
        )
        # 打开相机并获取相机内参
        self.zed.open(self.params)
        self.calib = self.zed.get_camera_information().camera_configuration.calibration_parameters

    # 获取RGB图像
    def get_rgb_images(self):
        left_image = sl.Mat()
        right_image = sl.Mat()

        while rclpy.ok():
            self.zed.grab()
            self.zed.retrieve_image(left_image, sl.VIEW.LEFT)
            self.zed.retrieve_image(right_image, sl.VIEW.RIGHT)

            left_rgb = cv2.cvtColor(left_image.get_data(), cv2.COLOR_BGRA2RGB)
            right_rgb = cv2.cvtColor(right_image.get_data(), cv2.COLOR_BGRA2RGB)

            left_msg = self.bridge.cv2_to_imgmsg(left_rgb, encoding="rgb8")
            right_msg = self.bridge.cv2_to_imgmsg(right_rgb, encoding="rgb8")

            self.left_pub.publish(left_msg)
            self.right_pub.publish(right_msg)


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = ZedMini()
        # 直接进入采集和发布循环
        node.get_rgb_images()
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.zed.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
