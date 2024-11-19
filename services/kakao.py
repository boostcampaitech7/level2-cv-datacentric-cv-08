import requests
import json

'''
KAKAO_ACCESS_KEY : 카카오톡 액세스 키가 저장된 json 형식 파일의 경로
'''

KAKAO_ACCESS_KEY = "KAKAO_ACCESS_KEY_PATH"

def get_access_token(token):
    '''
    카카오톡 API의 액세스 토큰을 가져오는 함수
    
    Parameter:
        token (str) : 액세스 키가 저장된 json 파일 경로
        
    Return:
        str : 액세스 토큰 문자열
    '''
    with open(token,'r') as fp:
        tokens = json.load(fp)
    return tokens.get('access_token')

def uuid_extract(token):
    '''
    카카오톡 친구 목록에서 각 친구의 UUID와 닉네임 추출 후 출력
    
    Parameter:
        token (str) : 카카오톡 API 액세스 토큰
        
    Return:
        각 친구의 UUID, 닉네임 콘솔에 출력
    '''
    url = "https://kapi.kakao.com/v1/api/talk/friends" #친구 목록 가져오기
    header = {"Authorization": 'Bearer ' + token}
    result = json.loads(requests.get(url, headers=header).text)
    friends_list = result.get("elements")
    for freind in friends_list:
        print(freind.get("uuid"),freind.get("profile_nickname"))

def send_message(receiver_uuids, message_text):
    '''
    카카오톡 메시지를 전송하는 함수.
    
    Parameters:
        receiver_uuids (list): 메시지를 받을 친구의 UUID 목록.
        message_text (str): 보낼 메시지의 텍스트 내용.
    
    Returns:
        dict: 메시지 전송 결과에 대한 응답 데이터.
    '''
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