"""Test the full prediction pipeline end-to-end."""
import requests
import time

print("Testing /api/companies...")
r = requests.get("http://127.0.0.1:8000/api/companies")
companies = r.json()["companies"][:3]
total = len(r.json()["companies"])
print(f"Found {total} companies")

ticker = companies[0]["ticker"]
print(f"\nTesting prediction for: {ticker}")
start = time.time()
r2 = requests.post(f"http://127.0.0.1:8000/api/predict/{ticker}")
elapsed = time.time() - start
print(f"Status: {r2.status_code} ({elapsed:.1f}s)")

if r2.status_code == 200:
    data = r2.json()
    print(f"\n=== RESULTS ===")
    print(f"Decision: {data.get('decision')}")
    print(f"Entry: {data.get('entry_price')}")
    print(f"SL: {data.get('stop_loss')} / TP: {data.get('take_profit')}")
    print(f"Risk:Reward: {data.get('risk_reward')}")
    print(f"Forecast: {data.get('forecast_horizon')}")
    print(f"Pipeline time: {data.get('pipeline_time_seconds')}s")
    print(f"\nIndicator report: {len(data.get('indicator_report', ''))} chars")
    print(f"Pattern report: {len(data.get('pattern_report', ''))} chars")
    print(f"Trend report: {len(data.get('trend_report', ''))} chars")
    print(f"Has pattern chart: {bool(data.get('pattern_chart'))}")
    print(f"Has trend chart: {bool(data.get('trend_chart'))}")
    print("\n=== SUCCESS! ===")
else:
    print(f"Error: {r2.text[:800]}")
