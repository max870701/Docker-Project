# 使用官方的 Python 容器
FROM python:3.9-slim

# 設置工作目錄
WORKDIR /app

# 複製應用代碼到容器中
COPY frontend.py /app

# 安裝依賴
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# 開放 8501 端口
EXPOSE 8501

# 啟動 Streamlit 應用
CMD ["streamlit", "run", "frontend.py", "--server.port", "8501", "--server.address", "0.0.0.0"]