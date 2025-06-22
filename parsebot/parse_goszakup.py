import requests
from bs4 import BeautifulSoup
import json
import time
import re

def clean_text(text):
    # Заменяем множественные пробелы и переносы на один перенос строки
    return '\n'.join(line.strip() for line in text.split('\n') if line.strip())

def extract_customer(text):
    match = re.search(r'Заказчик:\s*(.*?)(?=\s*$)', text)
    if match:
        return clean_text(match.group(1))
    return ""

def clean_subject(text):
    return clean_text(text.replace('История', '').strip())

def parse_lot_number(lot_text):
    parts = [part.strip() for part in lot_text.split('\n') if part.strip()]
    result = {
        'lot_id': parts[0] if len(parts) > 0 else "",
        'lot_name': parts[1] if len(parts) > 1 else "",
        'lot_status': parts[2] if len(parts) > 2 else "",
        'additional_info': parts[3:] if len(parts) > 3 else []
    }
    return result

def parse_goszakup(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        print(f"[INFO] Отправка запроса к URL: {url}")
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        print(f"[INFO] Получен ответ от сервера. Статус код: {response.status_code}")
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        table = soup.find('table', id='search-result')
        if not table:
            print("[ERROR] Таблица с id 'search-result' не найдена на странице")
            return None
            
        data = []
        rows = table.find_all('tr')
        print(f"[INFO] Найдено строк в таблице: {len(rows) - 1}")
        
        for idx, row in enumerate(rows[1:], 1):
            print(f"[INFO] Обработка строки {idx}/{len(rows) - 1}")
            cols = row.find_all('td')
            if len(cols) >= 6:
                lot_info = parse_lot_number(cols[0].text)
                announcement_text = cols[1].text
                subject = clean_subject(cols[2].text)
                quantity = clean_text(cols[3].text)
                amount = clean_text(cols[4].text)
                purchase_type = clean_text(cols[5].text)
                status = clean_text(cols[6].text) if len(cols) > 6 else ""
                
                customer = extract_customer(announcement_text)
                announcement = announcement_text.split('Заказчик:')[0].strip()
                
                announcement_link = cols[1].find('a')
                announcement_href = announcement_link.get('href') if announcement_link else ""
                
                subject_link = cols[2].find('a')
                subject_href = subject_link.get('href') if subject_link else ""
                
                data.append({
                    'lot_id': lot_info['lot_id'],
                    'announcement': lot_info['lot_name'],
                    'customer': lot_info['lot_status'],
                    'subject': subject,
                    'subject_link': subject_href,
                    'quantity': quantity,
                    'amount': amount,
                    'purchase_type': purchase_type,
                    'status': status
                })
                print(f"[INFO] Лот {lot_info['lot_id']} успешно обработан")
        
        print(f"[SUCCESS] Обработка страницы завершена. Получено лотов: {len(data)}")
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Ошибка при запросе: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] Произошла ошибка: {e}")
        return None

def save_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main(page=1, count_record=2000):
    print(f"\n[START] Запуск парсера для страницы {page} с количеством записей {count_record}")
    base_url = "https://goszakup.gov.kz/ru/search/lots"
    
    params = {
        'count_record': str(count_record),
        'page': str(page)
    }
    
    url = f"{base_url}?{requests.compat.urlencode(params)}"
    print(f"[INFO] Сформированный URL: {url}")
    
    data = parse_goszakup(url)
    
    if data is not None and len(data) > 0:
        print(f"[SUCCESS] Данные успешно получены со страницы {page}. Количество записей: {len(data)}")
        return data
    else:
        print(f"[WARNING] Нет данных на странице {page}")
        return []

if __name__ == "__main__":
    result = main() 
    if result:
        print(f"Всего получено записей: {len(result)}")
        save_to_json(result, 'goszakup_data.json')
        print("Данные сохранены в файл goszakup_data.json") 