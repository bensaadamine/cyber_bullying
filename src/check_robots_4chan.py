"""
check_robots_4chan.py — TP1 : why 4chan is NOT one of our sources
================================================================
4chan has an official read-only API, but its robots.txt forbids every robot
("User-agent: *  Disallow: /"). We respect it, so 4chan content only enters the
corpus indirectly, through the Reddit communities r/4chan and r/greentext.
"""

import requests

r = requests.get("https://a.4cdn.org/robots.txt", headers={"User-Agent": "TP-FR-student"})
print(r.status_code)
print(r.text)
print("4chan scraping allowed:", "Disallow: /\n" not in r.text + "\n")
