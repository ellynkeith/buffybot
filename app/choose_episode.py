from flask import Flask, redirect, render_template, request, url_for
import csv
import random
from collections import defaultdict

app = Flask(__name__)
app.secret_key = "FakeKey"

name = ""


@app.route("/", methods=["GET", "POST"])
@app.route("/index", methods=["GET", "POST"])
def index_page():
    global name
    if name and type(name) == str:
        name = choose_episode(name)
    if request.method == "POST":
        if "button" in request.form:
            return redirect(url_for("name_form_page"))
    return render_template(
        "index.html",
        name=name,
    )


@app.route("/user_info", methods=["GET", "POST"])
def name_form_page():
    global name
    if request.method == "POST":
        if "submit" in request.form:
            name = request.form.get("season")
            return redirect(url_for("index_page"))
        elif "home" in request.form:
            return redirect(url_for("index_page"))
    return render_template("name_form.html")


def process_episodes(episode_list):
    ep_dict = defaultdict(list)
    for line in episode_list:
        num, season, airdate, title = line.split("\t")
        season_key = season.split("-")[0]
        ep_key = num.replace(".", "")
        ep_dict[season_key].append({"season": season,
                                    "airdate": airdate,
                                    "title": title.replace("\n", "")
                                    })
    return ep_dict


def choose_episode(season=None):
    with open("episode_list.txt", "r") as f:
        lines = f.readlines()
    eps = process_episodes(lines)

    if season in eps.keys():
        try:
            choices = eps[season]
        except TypeError:
            import pdb; pdb.set_trace()
    else:
        choices = list()
        for season in eps:
            choices.extend(eps[season])

    return random.choice(choices)


if __name__ == "__main__":
    app.run()
