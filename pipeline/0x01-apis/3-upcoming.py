#!/usr/bin/env python3
"""
What will be next?
"""
import requests
# import pandas as pd


if __name__ == '__main__':
    url = "https://api.spacexdata.com/v4/launches/upcoming"

    r = requests.get(url).json()

    # data = pd.DataFrame.from_dict(r, orient='columns')
    # data.sort_values(by='date_unix', inplace=True)
    # # print(data)
    # name = data.iloc[0]['name']
    # date_local = data.iloc[0]['date_local']
    # rocket_id = data.iloc[0]['rocket']
    # launchpad_id = data.iloc[0]['launchpad']
    # rock = requests.get(
    #     'https://api.spacexdata.com/v4/rockets/' +
    #     rocket_id).json()
    # rocket_name = rock['name']

    # launchpad = requests.get(
    #     'https://api.spacexdata.com/v4/launchpads/' +
    #     launchpad_id).json()
    # launchpad_name = launchpad['name']
    # launchpad_locality = launchpad['locality']

    # print(
    #     "{} ({}) {} - {} ({})".format(
    #         name,
    #         date_local,
    #         rocket_name,
    #         launchpad_name,
    #         launchpad_locality))

    information = {
        'launch name': '',
        'date': '',
        'rocket id': '',
        "date_local": '',
        'launchpad id': ''}
    for i in r:
        date = i['date_unix']
        if information['date'] == '' or information['date'] > date:
            information['date'] = date
            information['launch name'] = i['name']
            information['rocket id'] = i['rocket']
            information['launchpad id'] = i['launchpad']
            information['date_local'] = i['date_local']
    rock = requests.get(
        'https://api.spacexdata.com/v4/rockets/' +
        information['rocket id']).json()
    rocket_name = rock['name']

    launchpad = requests.get(
        'https://api.spacexdata.com/v4/launchpads/' +
        information['launchpad id']).json()
    launchpad_name = launchpad['name']
    launchpad_locality = launchpad['locality']

    date = information['date_local']
    name = information['launch name']
    print(
        "{} ({}) {} - {} ({})".format(
            name,
            date,
            rocket_name,
            launchpad_name,
            launchpad_locality))
