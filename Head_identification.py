from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import cv2
import matplotlib.pyplot as plt
import numpy as np


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[: ,: ,i] = color_mask[i]
        ax.imshow(np.dstack((img, m* 0.35)))


def load_image(image_name):
    image_path = f'image/{image_name}.png'
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_model(model_path, model_type, device):
    sam_checkpoint = model_path
    model_type = model_type
    device = device
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    return sam


def process_image(image, sam):
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)

    # 将SAM识别结果掩码进行数字化处理，转存到masks中
    masks = [np.array(seg['segmentation'], dtype=np.uint8) for seg in masks]
    masks = np.stack(masks)

    return masks


def threshold_filtering(image, masks, threshold, image_name):
    # 使用颜色范围进行识别并生成 mask
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_color = np.array([100, 25, 20])  # 紫色的较低范围
    upper_color = np.array([180, 255, 255])  # 紫色的较高范围
    mask = cv2.inRange(hsv, lower_color, upper_color)

    filtered_masks = []

    for current_mask in masks:
        # Find the bounding box of the current mask
        rows, cols = np.where(current_mask == 1)
        min_row, max_row = np.min(rows), np.max(rows)
        min_col, max_col = np.min(cols), np.max(cols)

        region_mask = mask[min_row:max_row + 1, min_col:max_col + 1]
        region_current_mask = current_mask[min_row:max_row + 1, min_col:max_col + 1]

        overlap = np.sum((region_mask == 255) & (region_current_mask == 1))
        total_pixels = np.sum(region_current_mask == 1)
        overlap_percentage = (overlap / total_pixels) * 100

        # 置信参数，建议范围为(50,80)，值越高筛选越严格，容易漏选
        if overlap_percentage > threshold:
            filtered_masks.append(current_mask)

    filtered_masks = np.array(filtered_masks)

    # 返回最终生成的掩码
    return filtered_masks


def generate_multi_category_masks(filtered_masks, image_name):
    combined_masks_increment = np.zeros_like(filtered_masks[0], dtype=np.uint8)
    counter = 1

    for mask in filtered_masks:
        # Find the non-zero region of the current mask
        rows, cols = np.where(mask > 0)
        # If the region is empty, continue to the next mask
        if len(rows) == 0 or len(cols) == 0:
            continue
        # Extract the region from the combined mask for comparison
        min_row, max_row = np.min(rows), np.max(rows)
        min_col, max_col = np.min(cols), np.max(cols)
        region_combined = combined_masks_increment[min_row:max_row + 1, min_col:max_col + 1]
        region_current = mask[min_row:max_row + 1, min_col:max_col + 1]

        # Check if more than 10% of the region in the combined mask is non-zero
        if np.sum(region_combined > 0) / region_combined.size > 0.1:
            # If the current mask has fewer non-zero values in the region, skip it
            if np.sum(region_current) < np.sum(region_combined):
                continue

        # Increment the value of the current mask and add to the combined mask
        mask[mask > 0] = counter
        combined_masks_increment += mask
        counter += 1

        np.save(f'masks_{image_name}.npy', combined_masks_increment)

    return combined_masks_increment




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


def main(image_name, threshold, model_path="sam_vit_h_4b8939.pth", model_type="default", device="cuda"):
    # Load the image
    image = load_image(image_name)

    # Load the model
    sam = load_model(model_path, model_type, device)

    # Process the image
    masks = process_image(image, sam)

    filtered_masks = threshold_filtering(image, masks, threshold, image_name)

    combined_masks_increment = generate_multi_category_masks(filtered_masks, image_name)

    return combined_masks_increment




if __name__ == "__main__":
    image_name = "009"
    threshold = 60
    main(image_name, threshold)