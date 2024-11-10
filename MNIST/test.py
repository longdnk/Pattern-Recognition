import tensorflow as tf
import numpy as np
import argparse
import cv2
from tensorflow.keras.preprocessing import image


# Hàm tải mô hình đã lưu và nhận dạng ảnh
def load_and_predict(image_path, model_path="./model.keras"):
    # Tải mô hình
    model = tf.keras.models.load_model(model_path)

    # Đọc ảnh từ đường dẫn
    img = image.load_img(
        image_path, target_size=(28, 28), color_mode="grayscale"
    )  # Kích thước 28x28 cho MNIST

    # Chuyển ảnh thành mảng NumPy và chuẩn hóa
    img_array = image.img_to_array(img) / 255.0  # Chuẩn hóa giá trị pixel
    img_array = np.expand_dims(img_array, axis=0)  # Thêm chiều batch cho ảnh

    # Dự đoán
    predictions = model.predict(img_array)

    # Lấy lớp có xác suất cao nhất
    predicted_class = np.argmax(predictions, axis=1)

    # Chỉ hiển thị Predicted class
    print(f"Predicted class: {predicted_class[0]}")

    # Hiển thị hình ảnh và kết quả dự đoán bằng OpenCV
    img = cv2.imread(image_path)  # Đọc ảnh với OpenCV
    img_resized = cv2.resize(
        img, (300, 300)
    )  # Thay đổi kích thước ảnh để hiển thị rõ hơn
    cv2.putText(
        img_resized,
        f"Predicted: {predicted_class[0]}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )  # Vẽ kết quả dự đoán lên ảnh

    # Hiển thị ảnh bằng OpenCV
    while True:
        cv2.imshow("Predicted Image", img_resized)

        # Chờ phím bấm và kiểm tra nếu phím 'q' được nhấn
        key = cv2.waitKey(1)  # 1 ms để kiểm tra phím bấm
        if key == ord("q"):  # Nếu phím nhấn là 'q', thoát
            break

    cv2.destroyAllWindows()  # Đóng tất cả các cửa sổ OpenCV

    return predicted_class[0]


# Thiết lập parser cho dòng lệnh
def parse_args():
    parser = argparse.ArgumentParser(description="Tải mô hình và nhận dạng ảnh MNIST")
    parser.add_argument(
        "--image_path", type=str, help="Đường dẫn tới bức ảnh cần nhận dạng"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./model.keras",
        help="Đường dẫn tới mô hình đã lưu",
    )
    return parser.parse_args()


# Main function để chạy chương trình
def main():
    args = parse_args()

    # Sử dụng hàm load_and_predict với các tham số từ dòng lệnh
    load_and_predict(args.image_path, args.model_path)


if __name__ == "__main__":
    main()
