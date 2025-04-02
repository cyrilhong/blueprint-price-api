import logging
from flask import Flask, request, jsonify
from pycaret.regression import load_model, predict_model
import pandas as pd
import numpy as np
import traceback
from flask_cors import CORS
import os

# 設置日誌
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# 載入模型
model = load_model("models/final_model_with_pipeline")
logger.info("模型載入成功！")

def safe_divide(a, b, fill_value=0):
    """安全的除法運算"""
    return np.where(b > 0, a / b, fill_value)

def process_features(df):
    """處理輸入特徵，與訓練時保持一致"""
    # 確保數值欄位是數值型態
    numeric_cols = ['長度', '寬度', '高度', '靜壓mmAq', '馬力HP', 
                   '風量NCMM', '操作溫度°C', '採購數量', '葉輪直徑mm',
                   '風量效率', '功率密度']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 計算體積
    df['體積'] = df['長度'] * df['寬度'] * df['高度']
    
    # 基本特徵
    df['型號'] = df['品名'] + '_' + df['規格']  # 創建型號
    df['機殼材質'] = df['材質']  # 使用材質作為機殼材質
    df['架台材質'] = df['材質']  # 使用材質作為架台材質
    
    # 效率相關特徵
    df['壓力效率'] = safe_divide(df['靜壓mmAq'], df['馬力HP'])
    
    # 尺寸比例特徵
    df['長寬比'] = safe_divide(df['長度'], df['寬度'])
    df['高寬比'] = safe_divide(df['高度'], df['寬度'])
    
    # 確保所有必要的特徵都存在
    if '葉輪直徑mm' not in df.columns:
        df['葉輪直徑mm'] = 500  # 設定預設值
    if '功率密度' not in df.columns:
        df['功率密度'] = 0.00001  # 設定預設值
    if '風量效率' not in df.columns:
        df['風量效率'] = 20  # 設定預設值
    if '總金額' not in df.columns:
        df['總金額'] = 50000  # 設定預設值
    
    return df

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        logger.debug(f"收到的數據: {data}")
        
        # 處理輸入資料
        filtered_data = {}
        for key, value in data.items():
            filtered_data[key] = value[0] if isinstance(value, list) else value
        
        # 轉為 DataFrame
        input_df = pd.DataFrame([filtered_data])
        logger.debug(f"初始數據: {input_df.to_dict(orient='records')}")
        
        # 生成衍生特徵
        input_df = process_features(input_df)
        logger.debug(f"處理後的數據: {input_df.to_dict(orient='records')}")
        
        # 進行預測
        prediction = predict_model(model, data=input_df)
        predicted_price = prediction['prediction_label'].iloc[0]
        
        return jsonify({
            '預測價格': f"{predicted_price:,.2f} 元",
            '輸入資料摘要': {
                '尺寸': f"{filtered_data['長度']}x{filtered_data['寬度']}x{filtered_data['高度']} mm",
                '馬力': f"{filtered_data['馬力HP']} HP",
                '風量': f"{filtered_data['風量NCMM']} NCMM",
                '靜壓': f"{filtered_data['靜壓mmAq']} mmAq"
            }
        })
        
    except Exception as e:
        logger.error(f"預測錯誤: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            '錯誤': str(e),
            '訊息': '預測過程中發生錯誤',
            '除錯資訊': {
                '可用的特徵欄位': input_df.columns.tolist() if 'input_df' in locals() else None
            }
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)