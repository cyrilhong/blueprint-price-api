FROM python:3.9.7-slim

WORKDIR /app

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 複製必要檔案
COPY requirements.txt .
COPY models/ models/
COPY api.py .

# 建立虛擬環境並安裝套件
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 升級 pip 並安裝套件
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 設定環境變數
ENV PORT=8080

# 啟動命令
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "api:app"] 