#!/usr/bin/env python3
"""Build configs/cfb_teams.json: Polymarket US college-football team codes
joined to ESPN teams.

Polymarket's cfb slug codes are its own (aec-cfb-librty-coast-…); ESPN's
are different (LIB, CCU). Each Polymarket question names the schools in
away-vs-home order ("… event Liberty vs Coastal Carolina scheduled …"), which
matches ESPN's `location` field, so the join needs no hand-written table.
Re-run weekly (new opponents appear as the schedule rolls forward):

    python scripts/build_cfb_teams.py
"""
import json
import os
import re
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "configs", "cfb_teams.json")
GATEWAY = "https://gateway.polymarket.us/v1/markets?limit=500&active=true&closed=false&offset="
ESPN_TEAMS = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/teams?limit=1000"
Q = re.compile(r"football event (.+?) vs (.+?) scheduled", re.I)


# Polymarket question spelling -> ESPN `location` spelling
ALIASES = {
    "appalachian state": "app state", "central connecticut state": "central connecticut",
    "grambling state": "grambling", "louisiana-monroe": "ul monroe", "liu": "long island university",
    "miami (fl)": "miami", "penn": "pennsylvania", "rio grande": "rio grande",
    "southeastern louisiana": "se louisiana", "san jose state": "san josé state",
    "st. thomas (mn)": "st. thomas", "umass": "massachusetts", "ut rio grande valley": "ut rio grande valley",
}


def norm(s):
    s = s.lower().strip()
    s = ALIASES.get(s, s)
    s = s.replace("é", "e")
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def main():
    existing = {}
    if os.path.exists(OUT):
        existing = json.load(open(OUT))
    espn = {}
    d = json.load(urllib.request.urlopen(ESPN_TEAMS, timeout=30))
    for t in d["sports"][0]["leagues"][0]["teams"]:
        t = t["team"]
        rec = {"espn_abbr": t["abbreviation"].lower(), "location": t["location"], "display": t["displayName"]}
        espn.setdefault(norm(t["location"]), rec)
        espn.setdefault(norm(t["displayName"]), rec)
        espn.setdefault(norm(t.get("shortDisplayName", "")), rec)
        espn.setdefault(norm(t["abbreviation"]), rec)
    teams = dict(existing)
    off, seen = 0, 0
    while off < 60000:
        ms = json.load(urllib.request.urlopen(GATEWAY + str(off), timeout=30)).get("markets", [])
        if not ms:
            break
        for m in ms:
            slug = m.get("slug", "")
            if not slug.startswith("aec-cfb-"):
                continue
            parts = slug.split("-")
            if len(parts) != 7:
                continue
            q = Q.search(m.get("question", ""))
            if not q:
                continue
            seen += 1
            for code, loc in ((parts[2], q.group(1).strip()), (parts[3], q.group(2).strip())):
                e = espn.get(norm(loc)) or espn.get(norm(loc).replace(" state", " st"))
                entry = {"location": loc, "espn_abbr": e["espn_abbr"] if e else None,
                         "display": e["display"] if e else loc}
                if code not in teams or (entry["espn_abbr"] and not teams[code].get("espn_abbr")):
                    teams[code] = entry
        if len(ms) < 500:
            break
        off += 500
    unmatched = [c for c, v in teams.items() if not v.get("espn_abbr")]
    json.dump(teams, open(OUT, "w"), indent=1, sort_keys=True)
    print(f"markets parsed {seen}; teams {len(teams)}; without ESPN match {len(unmatched)}: {unmatched[:15]}")


if __name__ == "__main__":
    main()
