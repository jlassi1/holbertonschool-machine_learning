#!/usr/bin/env python3
"""
Where I am?
"""
import requests


def sentientPlanets():
    """
    method that returns the list of names
    of the home planets of all sentient species
    """
    r = requests.get("https://swapi-api.hbtn.io/api/species/")
    x = r.json()
    name_planet = []
    while True:
        for i in x["results"]:
            if 'sentient' in i.values():
                try:
                    m = requests.get(i["homeworld"]).json()
                    name_planet.append(m["name"])
                except Exception:
                    pass
        if x["next"] is None:
            break
        r = requests.get(x["next"])
        x = r.json()
    return name_planet
