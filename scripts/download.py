import requests
import os

def download_file(url, filename):
    file_dir = os.path.dirname(filename)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    if os.path.exists(filename):
        return
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(filename, 'wb') as file:
            file.write(response.content)
    except requests.RequestException as e:
        print(f"Error: {e}")

dir = "datasets"
print("download st-evcdp ...")
download_file('https://github.com/IntelligentSystemsLab/ST-EVCDP/raw/main/datasets/duration.csv', f'{dir}/duration.csv')
download_file('https://github.com/IntelligentSystemsLab/ST-EVCDP/raw/main/datasets/price.csv', f'{dir}/price.csv')
download_file('https://github.com/IntelligentSystemsLab/ST-EVCDP/raw/main/datasets/volume.csv', f'{dir}/volume.csv')
download_file('https://github.com/IntelligentSystemsLab/ST-EVCDP/raw/main/datasets/time.csv', f'{dir}/time.csv')

print("download urbanev ...")
download_file('https://github.com/IntelligentSystemsLab/UrbanEV/raw/main/data/e_price.csv', f'{dir}/urbanev/e_price.csv')
download_file('https://github.com/IntelligentSystemsLab/UrbanEV/raw/main/data/duration.csv', f'{dir}/urbanev/duration.csv')
download_file('https://github.com/IntelligentSystemsLab/UrbanEV/raw/main/data/volume.csv', f'{dir}/urbanev/volume.csv')

print("download ev-load-open-data ...")
names = ['acn', 'boulder_2021', 'palo_alto', 'sap', 'dundee', 'paris', 'perth']
for name in names:
    download_file(f'https://github.com/yvenn-amara/ev-load-open-data/raw/master/3.%20Output/{name}.csv', f'{dir}/{name}.csv')
