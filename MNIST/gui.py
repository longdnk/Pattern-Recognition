import tensorflow as tf
import numpy as np
import streamlit as st
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageDraw, ImageFont

# Tải mô hình
model_path = "./best_model.keras"
model = tf.keras.models.load_model(model_path)


def load_and_predict(image_path):
    # Đọc ảnh từ đường dẫn
    img = image.load_img(image_path, target_size=(28, 28), color_mode="grayscale")
    # Chuyển ảnh thành mảng NumPy và chuẩn hóa
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    # Dự đoán
    predictions = model.predict(img_array)
    # Lấy lớp có xác suất cao nhất
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class[0]


def process_single_image(uploaded_file):
    # Mở ảnh với PIL và hiển thị
    image_uploaded = Image.open(uploaded_file)

    # Gọi hàm dự đoán
    predicted_class = load_and_predict(uploaded_file)

    # Tạo hai cột để hiển thị hai ảnh cùng một lúc
    col1, col2 = st.columns(2)

    with col1:
        st.image(image_uploaded, caption="Uploaded images", use_container_width=True)

    with col2:
        # Chuyển ảnh sang RGB để có thể vẽ text màu
        img_rgb = image_uploaded.convert("RGB")
        # Resize ảnh để hiển thị tốt hơn
        img_resized = img_rgb.resize((300, 300))

        # Tạo ImageDraw object
        draw = ImageDraw.Draw(img_resized)

        font = ImageFont.truetype("Arial.ttf", 30)  # Tăng kích thước font

        # Vẽ text với màu xanh lá
        draw.text(
            (10, 10),  # Vị trí text
            f"Predicted: {predicted_class}",
            fill=(0, 255, 0),  # Màu xanh lá (RGB)
            font=font,
            stroke_fill=2,
            stroke_width=2
        )

        # Hiển thị ảnh đã được ghi text
        st.image(
            img_resized, caption="Image with predict", use_container_width=True
        )


def process_multiple_images(uploaded_files):
    for uploaded_file in uploaded_files:
        st.write(f"**Xử lý ảnh: {uploaded_file.name}**")
        # Tạo một divider để phân tách giữa các ảnh
        st.markdown("---")
        process_single_image(uploaded_file)


def main():
    st.title("MNIST Image Recognition")
    st.write("Upload image (28x28, grayscale) and receive result.")

    # Thêm radio button để chọn chế độ tải lên
    upload_mode = st.radio("Upload mode:", ("Single", "Multiple"))

    if upload_mode == "Single":
        uploaded_file = st.file_uploader("Select one", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            process_single_image(uploaded_file)

    else:  # Tải nhiều ảnh
        uploaded_files = st.file_uploader(
            "Select multiple", type=["png", "jpg", "jpeg"], accept_multiple_files=True
        )
        if uploaded_files:
            process_multiple_images(uploaded_files)


if __name__ == "__main__":
    main()
