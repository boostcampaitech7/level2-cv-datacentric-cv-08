import json
import requests

API_KEY = "KAKAO_RFRESH_KEY_PATH"
KAKAO_ACCESS_KEY = "KAKAO_ACCESS_KEY_PATH"

def api_key():
    with open(API_KEY, "r") as fp:
        key_data = json.load(fp)
    rest_api,refresh = key_data.get("rest_api_key"),key_data.get("refresh_token")
    return rest_api,refresh

def refresh_access_token():
    url = "https://kauth.kakao.com/oauth/token"
    
    REST_API_KEY,REFRESH_TOKEN = api_key()
    
    data = {
    "grant_type": "refresh_token",
    "client_id": REST_API_KEY,
    "refresh_token": REFRESH_TOKEN
    }
    
    response = requests.post(url, data=data)
    tokens = response.json()
    
    print(tokens)
    
    with open(KAKAO_ACCESS_KEY,'w') as fp:
        json.dump(tokens, fp)

refresh_access_token()