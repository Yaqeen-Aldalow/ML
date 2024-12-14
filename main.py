from fastapi import FastAPI
from generate_recommendations import get_recommendations  # استيراد الدالة من generate_recommendations.py

app = FastAPI()

# نقطة النهاية لتوليد التوصيات
@app.get("/recommendations/{user_id}")
async def get_movie_recommendations(user_id: int, n_recommendations: int = 10):
    """
    استرجاع التوصيات للمستخدم بناءً على ID المستخدم وعدد التوصيات المطلوبة.
    """
    # الحصول على التوصيات
    recommendations = get_recommendations(user_id, n_recommendations)
    
    # إعادة التوصيات في شكل JSON
    return {"user_id": user_id, "recommendations": recommendations}
