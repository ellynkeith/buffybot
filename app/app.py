from flask import Flask, render_template, request, jsonify, session
from buffybot import BuffyBot

app = Flask(__name__)
app.secret_key = 'your-secret-key'

# Initialize your bot
buffybot = BuffyBot()

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)