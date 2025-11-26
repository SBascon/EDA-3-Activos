import yfinance as yf
import pandas as pd

# Símbolos de los activos
assets = {"SP500": "^GSPC", "Gold": "GC=F", "BTC": "BTC-USD"}

for name, symbol in assets.items():
    # Descargar datos históricos de 60 meses con frecuencia mensual
    data = yf.download(symbol, period="5y", interval="1mo")
    
    # Guardar en CSV
    filename = f"{name}_60meses.csv"
    data.to_csv(filename)
    print(f"CSV guardado: {filename}")

