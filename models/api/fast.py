import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# from ML.model import load_model
from hitw.ML.model import load_model_local

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/predict")
def predict(le: float, eys: float, mys: float, gnipc: float, mmr: float,
            abr: float, co2_prod: float, mf: float, rpg: float, sub_region: str):
    """
    Make a single course prediction.
    Assumes `pickup_datetime` is provided as a string by the user in "%Y-%m-%d %H:%M:%S" format
    Assumes `pickup_datetime` implicitly refers to the "US/Eastern" timezone (as any user in New York City would naturally write)
    """

    dictionary = dict(le=le,
                    eys=eys,
                    mys=mys,
                    gnipc=gnipc,
                    mmr=mmr,
                    abr=abr,
                    co2_prod=co2_prod,
                    mf=mf,
                    rpg=rpg,
                    sub_region=sub_region)


    X_pred = pd.DataFrame(dictionary, index=[0])

    model = load_model_local()
    predict = model.predict(X_pred).tolist()
    return predict


@app.get("/")
def root():
    return {'greeting': 'Hello'}
