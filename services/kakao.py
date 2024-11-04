import requests
import json

KAKAO_ACCESS_KEY = "KAKAO_ACCESS_KEY_PATH"

def get_access_token(token):
    with open(token,'r') as fp:
        tokens = json.load(fp)
    return tokens.get('access_token')

def uuid_extract(token):
    url = "https://kapi.kakao.com/v1/api/talk/friends" #친구 목록 가져오기
    header = {"Authorization": 'Bearer ' + token}
    result = json.loads(requests.get(url, headers=header).text)
    friends_list = result.get("elements")
    for freind in friends_list:
        print(freind.get("uuid"),freind.get("profile_nickname"))

def send_message(receiver_uuids, message_text):
    access_token = get_access_token(KAKAO_ACCESS_KEY)
    url = "https://kapi.kakao.com/v1/api/talk/friends/message/default/send"
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type" : "application/x-www-form-urlencoded"
    }

    data = {
        'receiver_uuids' : json.dumps(receiver_uuids),
        'template_object': json.dumps({
            "object_type": "text",
            "text": message_text,
            "link": {
                "web_url": "",
                "mobile_web_url": ""
            },
            "button_title": ""
        })
    }
    response = requests.post(url, headers=headers, data=data)
    return response.json()