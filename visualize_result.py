from Head_identification import main
import cv2
import matplotlib.pyplot as plt
import numpy as np

def visualize_result(combined_masks_increment, image_name):
    # 从指定路径加载背景图片
    background_image = cv2.imread(f'image/{image_name}.png', cv2.IMREAD_COLOR)
    background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)

    # 获取 combined_masks_increment 中的所有类别标签
    unique_labels = np.unique(combined_masks_increment)

    # 创建一个颜色映射，为每个类别分配一个颜色（这里仅供示例，您可以根据需要更改）
    color_map = {
        label: np.random.randint(0, 256, 3) for label in unique_labels if label != 0
    }

    # 为每个类别创建一个彩色覆盖层
    overlay = np.zeros_like(background_image)
    for label, color in color_map.items():
        mask = (combined_masks_increment == label)
        overlay[mask] = color

        # 为每个类别的边界添加浅蓝色边线
        edges = cv2.Canny(mask.astype(np.uint8) * 255, 100, 200)
        edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        overlay[edges_dilated > 0] = [255, 191, 0]  # 外边框颜色

    # 将彩色覆盖层与背景图片合并
    blended_image = cv2.addWeighted(background_image, 1, overlay, 0.45, 0)

    # 使用matplotlib显示结果
    plt.imshow(blended_image)
    plt.axis('off')  # 隐藏坐标轴
    plt.show()




if __name__ == "__main__":

    image_name = "007"

    threshold = 60

    combined_masks_increment = main(image_name, threshold)

    # 可视化最终结果
    visualize_result(combined_masks_increment, image_name)
