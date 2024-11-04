import requests

def send_slack_notification(message):
    slack_webhook_url = "https://hooks.slack.com/services/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"  # Slack Webhook URL
    payload = {"text": message}
    requests.post(slack_webhook_url, json=payload)