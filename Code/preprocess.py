# preprocess.py
# import os
import cv2
import numpy as np
# from PIL import Image
# import io

class preprocess:
    """
    图像预处理类：
    - 图像增强（锐化 + 对比度增强）
    - 文件大小限制（>1MB 自动模糊压缩）
    """

    MAX_SIZE = 1 * 1024 * 1024  # 1MB

    def __init__(self):
        pass

    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        图像增强：锐化 + 对比度增强
        """
        # 锐化卷积核
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])

        sharpened = cv2.filter2D(image, -1, kernel)

        # 对比度增强
        enhanced = cv2.convertScaleAbs(sharpened, alpha=1.2, beta=10)

        return enhanced

    def blur_if_large(self, image: np.ndarray) -> np.ndarray:
        """
        若图像超过1MB，则进行高斯模糊压缩
        """
        is_success, buffer = cv2.imencode(".jpg", image)
        size = len(buffer)

        if size > self.MAX_SIZE:
            blurred = cv2.GaussianBlur(image, (25, 25), 0)
            return blurred
        return image

    def process(self, image_path: str) -> str:
        """
        处理图片并保存为临时文件
        返回处理后的图片路径
        """
        image = cv2.imread(image_path)
        image = self.enhance_image(image)
        image = self.blur_if_large(image)

        save_path = image_path.replace(".", "_processed.")
        cv2.imwrite(save_path, image)

        return save_path