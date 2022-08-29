import json
import time
import requests

crop_id = "ww8836961dbde0e9fa"
crop_agent_id = 1000002
crop_agent_secret = "eGn7caNHWBPUEf7BT7zUCB0OWsTCYahiU3crcAkN2Nc"


class WeChatPub:
    session = requests.session()

    def __init__(self):
        self.token = self.get_access_token(crop_id, crop_agent_secret, self.session)

    def get_access_token(self, crop_id, crop_agent_secret, session):
        url = f"https://qyapi.weixin.qq.com/cgi-bin/gettoken?corpid={crop_id}&corpsecret={crop_agent_secret}"
        rep = session.get(url)
        if rep.status_code != 200:
            print("request failed")
            return
        return json.loads(rep.content)['access_token']

    def send_msg(self, msg):
        print(self.token)
        content = "content test; ltz"
        url = "https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token=" + self.token

        data = {
            "touser": " LiZengGuang",
            "msgtype": "text",
            "agentid": 1000002,
            "text": {"content": content},
            "safe": "0"
        }
        rep = requests.post(url=url, json=data)
        print(rep.status_code)


if __name__ == "__main__":
    wechat = WeChatPub()
    s = requests.session()
    timenow = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    wechat.send_msg("msg")
    print('消息已发送！')