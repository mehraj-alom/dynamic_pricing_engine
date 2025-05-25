from fastapi import FastAPI, HTTPException
from .schemas import Item
import joblib
import pandas as pd
import os
import uvicorn

app = FastAPI()


MODEL_PATH = "app/model/lgbm_model.pkl"
try:
    model = joblib.load(MODEL_PATH)
except:
    raise RuntimeError("Model file not found!")


@app.post("/predict")
def predict(item: Item):
    try:
        if isinstance(item.Sizes, str):
            try:
                sizes_list = eval(item.Sizes)
                if not isinstance(sizes_list, list):
                    sizes_list = [item.Sizes]
            except:
                sizes_list = [s.strip().upper() for s in item.Sizes.split(",")]
        else:
            sizes_list = [item.Sizes]

        
        cleaned_sizes = [s.strip().upper().replace("LARGE", "L").replace("MEDIUM", "M").replace("SMALL", "S") for s in sizes_list]

      
        df = pd.DataFrame([{
            "BrandName": item.BrandName,
            "Category": item.Category,
            "MRP": item.MRP,
            "Details": item.Details,
            "Sizes_str": " ".join(cleaned_sizes)
        }])

        
        prediction = model.predict(df)[0]
        return {"predicted_price": round(prediction, 2)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    
    uvicorn.run(app, host="127.0.0.1", port=8000)