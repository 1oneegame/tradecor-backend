import requests
import pandas as pd
import json
from datetime import datetime, timedelta
import time

def parse_lots(page=1, per_page=500):
    base_url = "https://ows.goszakup.gov.kz/v3/lots"
    headers = {
        "Authorization": "Bearer YOUR_TOKEN"
    }
    
    params = {
        "limit": per_page,
        "page": page,
        "status": "published",
        "count": True
    }
    
    try:
        response = requests.get(base_url, headers=headers, params=params)
        response.raise_for_status()
        
        data = response.json()
        return data.get('items', []), data.get('total', 0)
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        return [], 0

def save_lots(lots):
    try:
        df = pd.DataFrame(lots)
        df.to_csv('suspicious_purchases.csv', index=False, mode='a', header=not pd.io.common.file_exists('suspicious_purchases.csv'))
        return True
    except Exception as e:
        print(f"Error saving data: {str(e)}")
        return False

def main(start_page=1, max_pages=None):
    all_lots = []
    page = start_page
    total_records = 0
    
    while True:
        lots, total = parse_lots(page=page)
        if not lots:
            break
            
        if total_records == 0:
            total_records = total
            
        all_lots.extend(lots)
        print(f"Parsed page {page}, got {len(lots)} lots. Total: {len(all_lots)}/{total_records}")
        
        if max_pages and page >= max_pages:
            break
            
        page += 1
        time.sleep(1)  # Задержка между запросами
        
    if all_lots:
        save_lots(all_lots)
        print(f"Successfully saved {len(all_lots)} lots")
    
    return all_lots

if __name__ == "__main__":
    main() 