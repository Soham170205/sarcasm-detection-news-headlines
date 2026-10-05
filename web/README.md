# Sarcasm Detector — Web Demo (React + Flask)

Place this `web/` folder inside your project root (next to `models/` and `src/`).

## 1. Backend (Flask) — terminal 1
```
cd web/backend
pip install -r requirements.txt
python app.py
```
Runs on http://127.0.0.1:5000

## 2. Frontend (React + Vite) — terminal 2
```
cd web/frontend
npm install
npm run dev
```
Open http://localhost:5173

## API
- POST /api/predict        {"headline": "..."}
- GET  /api/top-features?n=12
- GET  /api/history        (reads prediction_results.csv)
- DELETE /api/history
- GET  /api/model-info
- GET  /api/health
