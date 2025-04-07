import logging
from flask import Flask, request, jsonify
from pycaret.regression import load_model, predict_model
import pandas as pd
import numpy as np
import traceback
from flask_cors import CORS
import os
from dotenv import load_dotenv

# 根據環境載入不同的 .env 檔案
if os.environ.get('ENV') == 'production':
    load_dotenv('.env.production')
else:
    load_dotenv('.env.local')

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
    numeric_cols = [
        '長度', '寬度', '高度', '靜壓mmAq', '馬力HP', 
        '風量NCMM', '操作溫度°C', '採購數量'
    ]
    
    # 處理數值欄位
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if col in ['長度', '寬度', '高度']:
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna(df[col].median() if not df[col].empty else 0)
    
    # 計算衍生特徵
    df['體積'] = df['長度'] * df['寬度'] * df['高度']
    df['功率密度'] = np.where(df['體積'] > 0, df['馬力HP'] / df['體積'], 0)
    df['風量效率'] = np.where(df['馬力HP'] > 0, df['風量NCMM'] / df['馬力HP'], 0)
    df['壓力效率'] = np.where(df['馬力HP'] > 0, df['靜壓mmAq'] / df['馬力HP'], 0)
    df['長寬比'] = np.where(df['寬度'] > 0, df['長度'] / df['寬度'], 0)
    df['高寬比'] = np.where(df['寬度'] > 0, df['高度'] / df['寬度'], 0)
    
    # 確保類別特徵是字串類型
    categorical_cols = [
        "系列", "型號", "出口⽅向", "機殼材質", "架台材質",  # 移除 "規格"
        "產品名稱", "驅動方式", "防火花級", "單雙吸",
        "風機等級"
    ]
    
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].fillna('未知')
    
    # 如果需要將 "型號" 映射為 "規格"，在這裡處理
    if '型號' in df.columns:
        df['規格'] = df['型號']  # 新增這行，確保模型可以找到 "規格" 欄位
    
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
                '基本資訊': {
                    '尺寸': f"{filtered_data.get('長度')}x{filtered_data.get('寬度')}x{filtered_data.get('高度')} mm",
                    '規格': filtered_data.get('規格', '未知'),
                    '出口方向': filtered_data.get('出口⽅向', '未知')
                },
                '性能參數': {
                    '馬力': f"{filtered_data.get('馬力HP', '0')} HP",
                    '風量': f"{filtered_data.get('風量NCMM', '0')} NCMM",
                    '靜壓': f"{filtered_data.get('靜壓mmAq', '0')} mmAq"
                },
                '材質資訊': {
                    '機殼材質': filtered_data.get('機殼材質', '未知'),
                    '架台材質': filtered_data.get('架台材質', '未知')
                }
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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8081))
    app.run(host='0.0.0.0', port=port)