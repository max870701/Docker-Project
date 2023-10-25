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

# 安裝依賴
RUN pip install --no-cache-dir -r requirements.txt

# 啟動伺服器
CMD ["uvicorn", "api:app", "--reload"]