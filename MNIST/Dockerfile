FROM python:3.10-slim

# Đặt thư mục làm việc là /frontend
WORKDIR /app

# Sao chép tất cả các tệp và thư mục trừ src
COPY . .

# Cài đặt các dependencies
RUN pip install -r requirements.cpu.txt 

# Mở port 8501
EXPOSE 8501

# Đặt entrypoint để chạy Streamlit
ENTRYPOINT ["streamlit", "run", "gui.py"]