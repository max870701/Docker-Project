# 使用官方的 FastAPI 容器
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

# 設置工作目錄
WORKDIR /app

# 複製應用代碼到容器中
COPY api.py /app
COPY titanic-model.joblib /app
COPY requirements.txt /app

# 安裝依賴
RUN pip install --no-cache-dir -r requirements.txt

# 啟動伺服器
CMD ["python3", "api.py"]