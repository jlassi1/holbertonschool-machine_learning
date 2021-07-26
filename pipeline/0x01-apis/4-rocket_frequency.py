#!/usr/bin/env python3
"""
How many by rocket?
"""
import requests


if __name__ == '__main__':

    r = requests.get('https://api.spacexdata.com/v4/launches')
    launches = r.json()

    rock_launchs = {}
    for launch in launches:
        rocket_id = launch['rocket']
        j = requests.get(
            'https://api.spacexdata.com/v4/rockets/' + rocket_id).json()
        rocket_name = j['name']
        if rocket_name not in rock_launchs:
            rock_launchs[rocket_name] = 1
        else:
            rock_launchs[rocket_name] += 1

    for k, v in sorted(rock_launchs.items(), key=lambda k: k[1], reverse=True):
        print("{}: {}".format(k, v))
