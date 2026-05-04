import yfinance as yf

import pandas as pd

def load_equities(start, end):

    spx = yf.download("^GSPC", start=start, end=end, interval="1m")

    eu = yf.download("^STOXX50E", start=start, end=end, interval="1m")

    spx = spx.reset_index()[["Datetime", "Close"]].rename(

        columns={"Datetime": "timestamp", "Close": "spx"}

    )

    eu = eu.reset_index()[["Datetime", "Close"]].rename(

        columns={"Datetime": "timestamp", "Close": "eustoxx"}

    )

    # unify timezone

    spx["timestamp"] = pd.to_datetime(spx["timestamp"], utc=True)

    eu["timestamp"] = pd.to_datetime(eu["timestamp"], utc=True)

    return spx, eu
