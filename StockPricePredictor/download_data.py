import yfinance as yf

# Download all historical data for NASDAQ Composite
data = yf.download("^IXIC", start="1971-02-05")  # From NASDAQ's first trading day

# Save to CSV
data.to_csv("NASDAQ_Composite_Full_History.csv")
print("Data saved successfully!")