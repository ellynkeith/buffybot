import csv
import random
from collections import defaultdict
import sys


def process_episodes(episode_list):
	ep_dict = defaultdict(list)
	for line in episode_list:
		num, season, airdate, title = line.split("\t")
		season_key = season.split("-")[0]
		ep_key = num.replace(".", "")
		ep_dict[season_key].append({ 	"season": season,
										"airdate": airdate,
										"title": title.replace("\n", "") 
										})
	return ep_dict

def choose_episode(eps, season=None):
	if season:
		choices = eps[season]
	else:
		choices = list()
		for season in eps:
			choices.extend(eps[season])

	return random.choice(choices)


with open("episode_list.txt", "r") as f:
	lines = f.readlines()
episodes = process_episodes(lines)

if len(sys.argv) > 1:
	season = sys.argv[1]
else:
	season=None

this_one = choose_episode(episodes, season)
print(this_one)

