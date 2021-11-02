### buffybot
N-gram based random text generator using lines from Buffy the Vampire Slayer. (doesn't currently work because it's 5 years old and the corpus is long gone)

### choose_episode.py:
for when you want to watch Buffy but don't know what episode to watch. Filter by season or search all.
Usage:
```
(py3) ➜  buffybot git:(master) python choose_episode.py
{'season': '6-14', 'airdate': '12 Feb 02', 'title': 'Older and Far Away'}
(py3) ➜  buffybot git:(master) python choose_episode.py 3
{'season': '3-21', 'airdate': '18 May 99', 'title': 'Graduation Day: Part I'}
```

#### TODO:
* Rebuild corpus
* Exclude seasons or episodes
* Select based on main character, funny/sad rating


