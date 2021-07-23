#!/usr/bin/env python3
"""
Can I join?
"""
import requests


def availableShips(passengerCount):
    """
    method that returns the list of ships
    that can hold a given number of passengers
    """
    r = requests.get("https://swapi-api.hbtn.io/api/starships/")
    x = r.json()
    name_ship = []
    while x['next']:
        for i in x["results"]:
            if i["passengers"].isdigit():
                if int(i["passengers"]) > passengerCount:
                    name_ship.append(i["name"])

        r = requests.get(x["next"])
        x = r.json()
    return name_ship
