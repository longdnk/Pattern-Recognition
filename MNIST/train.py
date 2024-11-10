import tensorflow as tf
import numpy as np
import argparse
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    LayerNormalization,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)
from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd

GREEN = "\033[92m"
RESET = "\033[0m"


def plot_training_history(history):
    # Lấy dữ liệu từ đối tượng history
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    # Vẽ accuracy
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    # Vẽ loss
    plt.subplot(1, 2, 2)
    plt.plot(loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Hiển thị biểu đồ
    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.show()


def display_predictions(x_test, y_test, model, num_samples=20):
    predictions = model.predict(x_test[:num_samples])
    y_test_labels = np.argmax(y_test[:num_samples], axis=1)

    # Tính toán số lượng hàng và cột dựa trên số lượng mẫu
    rows = (num_samples + 3) // 4  # Làm tròn lên để tạo đủ số hàng
    cols = 5  # Số cột luôn là 4

    plt.figure(figsize=(12, 3 * rows))  # Tự động điều chỉnh chiều cao của figure
    for i in range(num_samples):
        plt.subplot(rows, cols, i + 1)  # Sử dụng rows và cols thay vì cố định 5x4
        plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
        plt.axis("off")
        true_label = y_test_labels[i]
        predicted_label = np.argmax(predictions[i])
        plt.title(f"True: {true_label} Pred: {predicted_label}")

    plt.tight_layout()
    plt.savefig("test_result.png")  # Lưu hình ảnh vào file
    plt.show()


def evaluate_model(model, x_test, y_test):
    # Dự đoán trên dữ liệu kiểm tra
    predictions = model.predict(x_test)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Ma trận nhầm lẫn
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Vẽ confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=np.arange(10),
        yticklabels=np.arange(10),
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()

    # Báo cáo phân loại (classification report)
    report = classification_report(y_true, y_pred, output_dict=True)
    print("\nClassification Report:")
    print(report)

    # Chuyển đổi classification report thành DataFrame
    report_df = pd.DataFrame(report).transpose()

    # Vẽ bảng classification report dưới dạng hình ảnh
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        report_df.iloc[:-1, :].T, annot=True, cmap="Blues", fmt=".2f", cbar=False
    )
    plt.title("Classification Report")
    plt.tight_layout()
    plt.savefig("classification_report_image.png")
    plt.show()

    # Lưu classification report vào CSV
    report_df.to_csv("classification_report.csv", index=True)  # Lưu vào file CSV

    # Precision, Recall, F1 Score (weighted average)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Vẽ Precision, Recall, F1 Score
    metrics = {"Precision": precision, "Recall": recall, "F1 Score": f1}
    plt.figure(figsize=(8, 6))
    plt.bar(metrics.keys(), metrics.values(), color=["green", "blue", "orange"])
    plt.title("Precision, Recall, F1 Score")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig("precision_recall_f1.png")
    plt.show()


def build_and_train_model(
    model_name: str,
    save_path: str,
    epochs: int,
    batch_size: int,
    multi_gpu: bool,
    num_samples_test: int,
):
    # Kiểm tra GPU
    if tf.config.list_physical_devices("GPU"):
        print(GREEN + "GPU Load" + RESET)
    else:
        print(GREEN + "CPU Load" + RESET)

    # Load và chuẩn hóa dữ liệu MNIST
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    INPUT_SHAPE = (28, 28, 1)
    KERNEL_SIZE = (3, 3)
    POOL_SIZE = (2, 2)

    # Sử dụng multi-GPU nếu được chỉ định
    if multi_gpu:
        strategy = tf.distribute.MirroredStrategy()
        print(
            GREEN
            + "Using {} GPUs for training".format(strategy.num_replicas_in_sync)
            + RESET
        )
    else:
        strategy = (
            tf.distribute.get_strategy()
        )  # Sử dụng single-GPU hoặc CPU nếu không có multi-GPU

    # Đảm bảo tất cả các thao tác liên quan đến mô hình nằm trong `strategy.scope`
    with strategy.scope():
        model = Sequential(
            [
                Conv2D(
                    32,
                    kernel_size=KERNEL_SIZE,
                    activation="relu",
                    input_shape=INPUT_SHAPE,
                ),
                MaxPooling2D(pool_size=POOL_SIZE),
                LayerNormalization(),
                Conv2D(64, kernel_size=KERNEL_SIZE, activation="relu"),
                MaxPooling2D(pool_size=POOL_SIZE),
                LayerNormalization(),
                Flatten(),
                Dense(128, activation="leaky_relu"),
                Dropout(0.5),
                Dense(10, activation="softmax"),
            ]
        )

        # Biên dịch mô hình trong `strategy.scope`
        model.compile(
            optimizer="adamw", loss="categorical_crossentropy", metrics=["accuracy"]
        )

    # Định nghĩa callback ModelCheckpoint để lưu mô hình tốt nhất
    checkpoint_callback = ModelCheckpoint(
        "best_model.keras",
        monitor="val_accuracy",  # Theo dõi validation accuracy
        save_best_only=True,  # Chỉ lưu khi có sự cải thiện
        mode="max",  # Lưu mô hình khi val_accuracy tăng lên
        verbose=1,
    )

    # Huấn luyện mô hình và lưu đối tượng history
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=[checkpoint_callback],  # Thêm callback vào quá trình huấn luyện
    )

    # Đánh giá mô hình trên dữ liệu kiểm tra
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(GREEN + f"Test accuracy: {test_acc:.4f}" + RESET)

    # In mô hình và lưu mô hình
    model.summary()
    model.save(save_path)

    # Vẽ đồ thị quá trình huấn luyện
    plot_training_history(history)

    # Hiển thị các dự đoán của 20 mẫu hình ảnh đầu tiên
    display_predictions(x_test, y_test, model, num_samples_test)

    # Đánh giá thêm các chỉ số như confusion matrix, precision, recall
    evaluate_model(model, x_test, y_test)

    print(GREEN + f"Plotting at '{model_name}_plot.png'" + RESET)


def parse_args():
    parser = argparse.ArgumentParser(description="Train and save a CNN model on MNIST")
    parser.add_argument(
        "--model_name", type=str, default="model", help="Tên mô hình (default: model)"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./model.keras",
        help="Đường dẫn lưu mô hình (default: ./model.keras)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Số lần epoch train (default: 10)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size (default: 128)"
    )
    parser.add_argument(
        "--num_samples_test", type=int, default=20, help="Batch size (default: 128)"
    )
    parser.add_argument(
        "--multi_gpu",
        type=bool,
        default=False,
        help="Sử dụng multi-GPU nếu True (default: False)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_and_train_model(
        model_name=args.model_name,
        save_path=args.save_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        multi_gpu=args.multi_gpu,
        num_samples_test=args.num_samples_test,
    )
