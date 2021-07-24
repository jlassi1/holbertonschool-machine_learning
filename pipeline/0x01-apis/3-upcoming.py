#!/usr/bin/env python3
"""
What will be next?
"""
from os import name
import requests
from datetime import datetime
import pandas as pd


if __name__ == '__main__':
    url = "https://api.spacexdata.com/v4/launches/upcoming"

    r = requests.get(url).json()
    # dataframe = pd.DataFrame.from_dict(r, orient='columns')
    # print(dataframe.columns)

    information = {'launch name': '', 'date': '', 'rocket id': '',
                   'launchpad id': ''}
    for i in r:
        lan_name = i['name']
        date = i['date_unix']
        rock_id = i['rocket']
        lanpad_id = i['launchpad']
        if information['date'] == '' or information['date'] > date:
            information['date'] = date
            information['launch name'] = lan_name
            information['rocket id'] = rock_id
            information['launchpad id'] = lanpad_id
    rock = requests.get(
        'https://api.spacexdata.com/v4/rockets/' +
        information['rocket id']).json()
    rocket_name = rock['name']

    launchpad = requests.get(
        'https://api.spacexdata.com/v4/launchpads/' +
        information['launchpad id']).json()
    launchpad_name = launchpad['name']
    launchpad_locality = launchpad['locality']

    date = datetime.fromtimestamp(information['date'])
    name = information['launch name']
    print(
        '{} ({}) {} - {} ({})'.format(
            name,
            date,
            rocket_name,
            launchpad_name,
            launchpad_locality))
