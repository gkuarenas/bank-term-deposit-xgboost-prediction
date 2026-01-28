'''
FASTAPI + GRADIO SERVING APPLICATION
====================================
'''
from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
from src.serving.inference import predict

# ===== Initialize FastAPI App ===== #
app = FastAPI(
    title="Bank Client Term Deposit Subscription Prediction API",
    description="ML API for predicting whether a client subscribes to a bank term teposit.",
    version="1.0.0"
)

@app.get("/")
def root():
    '''
    Health check for monitoring and load balancer.
    '''
    return{"status": "ok"}

# ===== REQUEST DATA SCHEMA ===== #
# Pydantic model for automatic validation and API documentation
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

# ===== MAIN PREDICTION API ENDPOINT ===== #
@app.post("/predict")
def get_prediction(data: CustomerData):
    """
    Main prediction endpoint for bank term deposit subscription prediction

    This endpoint:
    1. Receives validated client data via Pydantic model
    2. Calls the inference pipeline to transform features and predict
    3. Returns subscription prediction in JSON format

    Expected Response:
    - {"prediction": "Likely to subscribe"} or {"prediction": "Not likely to subscribe"}
    - {"error": "error_message"} if prediction fails
    """

    try:
        # Convert Pydantic model to dict and call inference pipeline
        result = predict(data.dict())
        return {"prediction": result}
    except Exception as e:
        # Return error details for debugging (Consider logging in production)
        return {"error": str(e)}
    
# ===== GRADIO WEB INTERFACE ===== #

def gradio_interface(
        age, job, marital, education, default, balance,
        housing, loan, contact, day, month, duration,
        campaign, pdays, previous, poutcome
):
    """
    This is the Gradio interface for this app

    This function: 
    1. Takes individual form inputs from Gradio UI
    2. Constructs the data dictionary matching the API schema
    3. Calls the same inference pipeline used by the API
    4. Returns user-friendly prediction string 

    """

    data = {
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

    result = predict(data)
    return str(result) # Return as string for Gradio display

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
    theme = gr.themes.Soft()
)

# ===== MOUNT GRADIO INTO FASTAPI ===== #
app = gr.mount_gradio_app(
    app,            # FastAPI application instance
    demo,           # Gradio interface
    path="/ui"      # URL path where Gradio will be accessible
)