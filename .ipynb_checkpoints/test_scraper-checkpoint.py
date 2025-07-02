import requests
from bs4 import BeautifulSoup

URL = "https://www.lacentrale.fr/auto-occasion-annonce-69116060022.html"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

print(f"Download : {URL}")
response = requests.get(URL, headers=HEADERS)

if response.status_code == 200:
    print("Success !")
    soup = BeautifulSoup(response.content, 'html.parser')
    
    car_details = {}

    # --- Selectors ---

    # 1. PRICE
    
    price_element = soup.find('div', attrs={'data-testid': 'vehicle-price'})
    car_details['price'] = price_element.text.strip() if price_element else "Prix non trouvé"

    # 2. TITLE (brand and model)
    title_element = soup.find('h1')
    title_text = title_element.text.strip() if title_element else ""
    car_details['title'] = title_text
    
    if title_text:
        parts = title_text.split(' ')
        car_details['brand'] = parts[0]
        car_details['model'] = " ".join(parts[1:]) if len(parts) > 1 else ""
    else:
        car_details['brand'] = "Brand not found"
        car_details['model'] = "Model not found"

    # 3. CARACTÉRISTICS (Year, Km, etc.)
    labels = soup.find_all('div', class_='SummaryInformation_label')
    # All the values  (2023, 10 km...) are in 'div' with another class.
    values = soup.find_all('div', class_='SummaryInformation_value')
    
    # Combining to create key-value
    if len(labels) == len(values):
        for i in range(len(labels)):
            key = labels[i].text.strip().lower().replace(' ', '_').replace(':', '')
            value = values[i].text.strip()
            car_details[key] = value
    else:
        print("Warning.")


    # --- Affichage des résultats ---
    print("\n--- Car's info ---")
    for key, value in car_details.items():
        print(f"{key.capitalize().replace('_', ' ')} : {value}")
    
else:
    print(f"Download failed. Code de statut : {response.status_code}")