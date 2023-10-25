# 使用官方的 FastAPI 容器
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

# 設置工作目錄
WORKDIR /app

# 創建 Models 存放目錄
RUN mkdir models

# 複製應用代碼到容器中
COPY api.py /app
COPY requirements.txt /app
COPY preprocessing.py /app

# 開放 8000 端口
EXPOSE 8000

# 安裝依賴
RUN pip install --no-cache-dir -r requirements.txt

# 啟動伺服器
CMD ["uvicorn", "api:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]