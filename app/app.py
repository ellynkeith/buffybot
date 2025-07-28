from flask import Flask, render_template, request, jsonify, session, send_file
from buffybot import BuffyBot, BASE_DIR
from datetime import datetime
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

app = Flask(__name__)
app.secret_key = 'your-secret-key'

FEEDBACK_DATA_PATH = BASE_DIR / 'feedback/feedback_data.csv'

# Initialize your bot
buffybot = BuffyBot()

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/download-feedback')
def download_feedback():
    if not os.path.exists(FEEDBACK_DATA_PATH):
        return "No feedback data found yet", 404

    password = request.args.get('password')
    if password != 'buffybot2025':
        return "Access denied", 403

    date_str = datetime.now().strftime('%Y-%m-%d')
    download_name = f'buffy_feedback_{date_str}.csv'

    return send_file(FEEDBACK_DATA_PATH, as_attachment=True, download_name=download_name)


@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    if not os.path.exists(FEEDBACK_DATA_PATH):
        feedback_df = pd.DataFrame(columns=list(data.keys()))
    else:
        feedback_df = pd.read_csv(FEEDBACK_DATA_PATH)

    try:
        new_row = pd.DataFrame([data])
        feedback_df = pd.concat([feedback_df, new_row], ignore_index=True)
        feedback_df.to_csv(FEEDBACK_DATA_PATH, index=False)

        return jsonify({'status': 'success'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data['message']
    character = data['character']
    print('current session: {}'.format(session))

    # Get or create session_id
    if 'session_id' not in session or data['character'] != session['character']:
        # Create new session for new character
        conversation = buffybot.start_conversation(data['character'])
        session['session_id'] = conversation['session_id']

        if 'conversation_history' not in session:
            session['conversation_history'] = []

        # Replay previous conversation
        for exchange in session.get('conversation_history', []):
            buffybot.add_to_history(
                session['session_id'],
                exchange['user'],
                exchange['assistant']
            )
        print(f'starting conversation with {character}')
        conversation = buffybot.start_conversation(character)
        session['session_id'] = conversation['session_id']
        session['character'] = character
        print(f'session id is {conversation["session_id"]}')

    print(f'fetching response from {character}, session {session["session_id"]}')
    result = buffybot.chat(session['session_id'], user_message)

    session['conversation_history'].append({
        'user': user_message,
        'assistant': result['response'],
        'character': character
    })

    if 'error' in result:
        return jsonify({'error': result['error']})
    else:
        return jsonify({'response': result})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    is_production = 'PORT' in os.environ
    app.run(
        host='0.0.0.0' if is_production else '127.0.0.1',
        port=port,
        debug=not is_production
    )