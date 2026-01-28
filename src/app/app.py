from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
import pathlib as Path
import sys

# Ensure we can import from src/serving when running "uvicorn src.app.app:app"
sys.path.append(str(Path(__file__).resolve().parent.parent))

from serving.inference import predict

app = FastAPI()

@app.get("/")
def root():
    return {"status": "ok"}

class CustomerData(BaseModel):
    # Bank CLient Data
    age: int            
    job: str
    marital: str
    education: str  
    default: str
    balance: int
    housing: str
    loan: str

    # Last Contact of Client
    contact: str
    day: int
    month: str
    duration: int

    # Other attributes
    campaign: int
    pdays: int
    previous: int
    poutcome: str

@app.post("/predict")
def api_predict(data: CustomerData):
    try:
        out = predict(data.dict())
        return {"prediction": out}
    except Exception as e:
        return {"error": str(e)}

def gradio_interface(
        age, job, marital, education, default, balance,
        housing, loan, contact, day, month, duration,
        campaign, pdays, previous, poutcome
):
    payload = {
        "age": int(age),
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "balance": int(balance),
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "day": int(day),
        "month": month,
        "duration": int(duration),
        "campaign": int(campaign), 
        "pdays": int(pdays),
        "previous": int(previous),
        "poutcome": poutcome
    }

    out = predict(payload)
    return str(out)

demo = gr.Interface(
    fn=gradio_interface,
    inputs = [
        # ===== Client profile (demographics / finance) =====
        gr.Number(label="Age", value=35, minimum=18, maximum=120, step=1),
        gr.Dropdown(
            [
                "admin.", "unknown", "unemployed", "management", "housemaid",
                "entrepreneur", "student", "blue-collar", "self-employed",
                "retired", "technician", "services"
            ],
            label="Job", value="management"
        ),
        gr.Dropdown(["married", "divorced", "single"], label="Marital Status", value="single"),
        gr.Dropdown(["unknown", "secondary", "primary", "tertiary"], label="Education", value="tertiary"),
        gr.Dropdown(["yes", "no"], label="Credit in Default?", value="no"),
        gr.Number(label="Average Yearly Balance (€)", value=1000, minimum=-100000, maximum=1000000, step=1),
        gr.Dropdown(["yes", "no"], label="Has Housing Loan?", value="no"),
        gr.Dropdown(["yes", "no"], label="Has Personal Loan?", value="no"),

        # ===== Last contact (current campaign) =====
        gr.Dropdown(["unknown", "telephone", "cellular"], label="Contact Type", value="cellular"),
        gr.Number(label="Last Contact Day of Month", value=15, minimum=1, maximum=31, step=1),
        gr.Dropdown(
            ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"],
            label="Last Contact Month", value="may"
        ),
        gr.Number(label="Last Contact Duration (seconds)", value=120, minimum=0, maximum=10000, step=1),

        # ===== Campaign history =====
        gr.Number(label="Campaign Contacts (this campaign)", value=1, minimum=1, maximum=1000, step=1),
        gr.Number(
            label="Pdays (days since last contact; -1 = not previously contacted)",
            value=-1, minimum=-1, maximum=10000, step=1
        ),
        gr.Number(label="Previous Contacts (before this campaign)", value=0, minimum=0, maximum=1000, step=1),
        gr.Dropdown(["unknown", "other", "failure", "success"], label="Previous Outcome", value="unknown"),
    ],
    outputs = gr.Textbox(label="Term Deposit Prediction", lines=2),
    title = "Bank Term Deposit Predictor",
    description = "Fill in client details + last contact details to predict subscription.",
)

app = gr.mount_gradio_app(app, demo, path="/ui")