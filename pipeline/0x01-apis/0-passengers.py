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
    while True:
        for i in x["results"]:
            if i["passengers"].isdigit() or "," in i["passengers"]:
                num = int(i["passengers"].replace(',', ''))
                if num >= passengerCount:
                    name_ship.append(i["name"])

        if x["next"] is None:
            break
        r = requests.get(x["next"])
        x = r.json()
    return name_ship
