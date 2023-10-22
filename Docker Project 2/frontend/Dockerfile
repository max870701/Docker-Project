# 使用官方的 Python 容器
FROM python:3.9-slim

# 設置工作目錄
WORKDIR /app

# 複製應用代碼到容器中
COPY frontend.py /app

# 安裝依賴
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# 啟動 Streamlit 應用
CMD ["streamlit", "run", "frontend.py"]
