import json
import requests

'''
API_KEY : 카카오톡 리프레쉬, REST API 키가 저장된 json 형식의 파일의 경로
KAKAO_ACCESS_KEY : 카카오톡 액세스 키가 저장된 json 형식 파일의 경로
'''

API_KEY = "KAKAO_REFRESH_KEY_PATH"
KAKAO_ACCESS_KEY = "KAKAO_ACCESS_KEY_PATH"

def api_key():
    '''
    카카오톡 API의 REST API키와 리프레쉬 토큰을 가져오는 함수
    
    Return:
        tuple: REST API 키와 리프레쉬 토큰
    '''
    with open(API_KEY, "r") as fp:
        key_data = json.load(fp)
    rest_api,refresh = key_data.get("rest_api_key"),key_data.get("refresh_token")
    return rest_api,refresh

def refresh_access_token():
    '''
    카카오톡 API의 리프레쉬 토큰을 사용하여 액세스 토큰을 갱신하는 함수.
    새롭게 발급된 액세스 토큰을 JSON 형식으로 저장
    
    Returns:
        None
    '''
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