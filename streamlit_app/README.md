# Streamlit UI (add-on)

A simple web UI on top of the FastAPI churn prediction backend, so anyone can
test the model by filling in a form instead of using `/docs` or `curl`.

This is a thin client — it does not load or run the model itself. It sends
the form values to the FastAPI backend's `/predict` endpoint and displays
the response.

## Running locally

You need the FastAPI backend running first (see the main project README):

```bash
# In one terminal, from the project root:
uvicorn app.main:app --port 8000

# In a second terminal:
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

## Pointing it at a deployed API

By default this UI looks for the API at `http://localhost:8000`. Once
you've deployed the FastAPI backend (e.g. to Render), point the UI at it
instead by setting the `API_URL` environment variable:

```bash
API_URL=https://your-churn-api.onrender.com streamlit run app.py
```

## Deploying this UI to Streamlit Community Cloud

1. Push this project to GitHub (if not already done).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **"New app"**, select this repo, and set the main file path to `streamlit_app/app.py`.
4. Under **Advanced settings → Secrets**, add:
   ```toml
   API_URL = "https://your-churn-api.onrender.com"
   ```
   (this points the UI at your deployed FastAPI backend, not localhost)
5. Deploy. You'll get a public `*.streamlit.app` URL.

Note: your FastAPI backend needs to be deployed and running (e.g. on Render)
*before* this UI will return real predictions — this UI has no model of its
own, it's purely a front-end for the API.
