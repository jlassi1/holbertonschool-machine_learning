#!/usr/bin/env python3
"""
Rate me is you can!
"""
import sys
from datetime import datetime
import requests


if __name__ == '__main__':
    url = sys.argv[1]
    try:
        r = requests.get(url)
        if r.status_code == 200:
            print(r.json()['location'])
        if r.status_code == 404:
            print('Not found')
        elif r.status_code == 403 or r.status_code == 429:
            limit = datetime.fromtimestamp(
                int(r.headers['X-RateLimit-Reset'])).minute
            now = datetime.now().minute
            print("Reset in {} min".format(limit-now))
    except Exception:
        print('Not found')
