import base64
import requests
import time

source_paths = ['./tests/images/family-parents-grandparents-Morsa-Images-Taxi-56a906ad3df78cf772a2ef29.jpg']
# source_paths = ['./tests/images/elon-1.jpg']
target_path = './tests/images/mark.jpg'


def image_to_base64_str(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        return encoded_string.decode('utf-8')


def request(source_paths, target_path):
    sources = []
    for source_path in source_paths:
        sources.append(image_to_base64_str(source_path))
    target = image_to_base64_str(target_path)
    params = {
        'source': image_to_base64_str(source_path),
        'target': target,
        'source_extension': '.jpg',
        'target_extension': '.jpg'
    }
    url = 'http://127.0.0.1:3062/check-face'
    response = requests.post(url, json=params)
    print("Status Code:", response.status_code)
    if response.status_code == 200:
        output_data = base64.b64decode(response.json()['output'])
        with open(f'output/{int(time.time())}.jpg', 'wb') as f:
            f.write(output_data)
    else:
        print("Error: The request did not succeed.")


request(source_paths, target_path)