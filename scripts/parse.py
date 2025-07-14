from bs4 import BeautifulSoup
import pandas as pd
import re
from pathlib import Path
import openai
import json
import time
from typing import List, Dict

BUFFY_CHARACTERS = {
    # Main characters
    'BUFFY', 'WILLOW', 'XANDER', 'GILES', 'CORDELIA', 'OZ', 'ANGEL', 'SPIKE', 'TARA',
    'ANYA', 'RILEY', 'JOYCE', 'DAWN', 'FAITH', 'WESLEY', 'JONATHAN', 'ANDREW', 'WARREN',

    # Recurring characters
    'PRINCIPAL WOOD', 'ROBIN', 'KENNEDY', 'CALEB', 'GLORY', 'BEN', 'ADAM', 'MAGGIE',
    'PROFESSOR WALSH', 'FORREST', 'GRAHAM', 'HARMONY', 'DRUSILLA', 'DARLA', 'MASTER',
    'MAYOR', 'RICHARD WILKINS', 'WILKINS', 'SNYDER', 'PRINCIPAL SNYDER', 'JENNY',
    'MS CALENDAR', 'JENNY CALENDAR', 'KENDRA', 'LARRY', 'DEVON', 'PARKER', 'KATE',
    'HALFREK', 'D\'HOFFRYN', 'CLEM', 'SWEET', 'FIRST', 'THE FIRST', 'UBERVAMP',

    # Parents/family
    'HANK', 'SHEILA', 'SHEILA ROSENBERG', 'IRA ROSENBERG', 'JESSICA HARRIS',
    'TONY HARRIS', 'MR GILES', 'OLIVIA',

    # School/Sunnydale residents
    'FLUTIE', 'PRINCIPAL FLUTIE', 'WHITMORE', 'MAGGIE WALSH', 'MACHIDA', 'SID',
    'AMPATA', 'MARCIE', 'INVISIBLE GIRL', 'TODD', 'LANCE', 'SCOTT', 'SCOTT HOPE',
    'OWEN', 'RODNEY', 'PERCY', 'MICHAEL', 'CASSIE', 'HOLDEN', 'AMANDA', 'VI',
    'RONA', 'MOLLY', 'SHANNON', 'CARIDAD', 'CHAO-AHN',

    # Demons/monsters/misc
    'ANYANKA', 'VENGEANCE DEMON', 'WISH VENGEANCE DEMON', 'BALTHAZAR', 'KAKISTOS',
    'MR TRICK', 'TRICK', 'LUKE', 'COLIN', 'JESSE', 'THOMAS', 'MACHIDA REPTILE',
    'SHED GUY', 'EGG BOY', 'HYENA STUDENT', 'ROBOT', 'BUFFYBOT', 'APRIL',
    'TED', 'MOM', 'DAD', 'CLERK', 'WAITRESS', 'DOCTOR', 'NURSE', 'COP',
    'POLICE OFFICER', 'OFFICER', 'VAMPIRE', 'DEMON', 'STUDENT', 'GIRL', 'BOY',
    'MAN', 'WOMAN', 'VOICE', 'ANNOUNCER', 'REPORTER'
}


def parse_blockquote_format(blockquote_element, episode_num, episode_title):
    """Parse the terrible Season 7 blockquote format"""
    dialogue_data = []
    current_scene = "Unknown"

    # Get elements recursively
    all_elements = blockquote_element.find_all(['p', 'b', 'br'])

    i = 0
    while i < len(all_elements):
        element = all_elements[i]

        if element.name == 'b':
            text = clean_text(element.get_text())
            print(f"DEBUG: Found <b> tag: '{text}'")

            # Check if this looks like a character name
            if is_valid_character(text):
                character = text.upper()
                print(f"DEBUG: Valid character: {character}")

                # The dialogue might be in the parent <p> tag, after this <b> tag
                parent_p = element.find_parent('p')
                if parent_p:
                    # Get all text from the <p> tag, excluding the <b> character name
                    full_text = parent_p.get_text()
                    # Remove the character name from the beginning
                    dialogue_text = full_text.replace(text, '', 1).strip()

                    if dialogue_text and len(dialogue_text) > 3:
                        dialogue = clean_dialogue(dialogue_text)
                        if dialogue:
                            dialogue_data.append({
                                'episode_num': episode_num,
                                'episode_title': episode_title,
                                'scene': current_scene,
                                'character': character,
                                'dialogue': dialogue,
                                'is_voiceover': 'VO' in text or 'VOICEOVER' in text
                            })

        i += 1

    print(f"DEBUG: Total dialogue lines found: {len(dialogue_data)}")
    return dialogue_data


def parse_pre_format(pre_element, episode_num, episode_title):
    """Parse <pre> format transcripts"""
    dialogue_data = []

    # Get all text and split by lines
    text_content = pre_element.get_text()
    lines = text_content.split('\n')

    current_scene = "Unknown"

    for line in lines:
        line = clean_text(line)
        if not line:
            continue

        # Check if it's dialogue (Character: text)
        dialogue_match = re.match(r'^([A-Za-z][A-Za-z\s]+?):\s*(.+)', line)
        if dialogue_match:
            character = dialogue_match.group(1).strip().upper()
            if is_valid_character(character):
                raw_dialogue = dialogue_match.group(2).strip()
                dialogue = clean_dialogue(raw_dialogue)

                if dialogue:
                    dialogue_data.append({
                        'episode_num': episode_num,
                        'episode_title': episode_title,
                        'scene': current_scene,
                        'character': character,
                        'dialogue': dialogue,
                        'is_voiceover': 'VOICEOVER' in character
                    })
        else:
            # Check if it's a scene description we want to track
            if len(line) < 100 and any(word in line.lower() for word in
                                       ['night', 'day', 'graveyard', 'school', 'library', 'bronze', 'buffy\'s room']):
                current_scene = line[:50]  # Truncate long scene descriptions

    return dialogue_data

def is_valid_character(character_name):
    """Check if a character name is in our approved list"""
    # Clean up the character name
    cleaned = character_name.strip().upper()

    # Remove common suffixes that might be added
    cleaned = re.sub(r'\s+(VOICEOVER|VO)$', '', cleaned)

    return cleaned in BUFFY_CHARACTERS

def clean_text(text):
    """Clean text of HTML entities and normalize whitespace"""
    # Replace non-breaking spaces and other HTML entities
    text = text.replace('\xa0', ' ')
    # Normalize whitespace and newlines
    text = ' '.join(text.split())
    return text.strip()

def clean_dialogue(dialogue):
    cleaned = re.sub(r'^\s*\([^)]+\)\s*', '', dialogue)
    return cleaned.strip()

def parse_dialogue_block(element, episode_num, episode_title):
    """Parse a block that might contain multiple dialogue lines separated by <br> tags"""
    dialogue_lines = []

    # Get the HTML content and split on <br> tags
    html_content = str(element)

    # Split on <br/> or <br> tags
    parts = re.split(r'<br\s*/?>', html_content)

    for part in parts:
        # Clean HTML tags and entities
        soup_part = BeautifulSoup(part, 'html.parser')
        text = clean_text(soup_part.get_text())

        if not text:
            continue

        # Check if this looks like dialogue
        dialogue_match = re.match(r'^([A-Za-z][A-Za-z\s]+?):\s*(.+)', text)
        if dialogue_match:
            character = dialogue_match.group(1).strip().upper()
            raw_dialogue = dialogue_match.group(2).strip()
            dialogue = clean_dialogue(raw_dialogue)

            if is_valid_character(character):
                raw_dialogue = dialogue_match.group(2).strip()
                dialogue = clean_dialogue(raw_dialogue)

                if dialogue:
                    # Only add if there's actual dialogue
                    dialogue_lines.append({
                        'episode_num': episode_num,
                        'episode_title': episode_title,
                        'character': character,
                        'dialogue': dialogue,
                        'is_voiceover': 'VOICEOVER' in character.upper()
                    })
    return dialogue_lines

def parse_buffy_transcript(html_file):
    with open(html_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    soup = BeautifulSoup(content, 'html.parser')

    title_tag = soup.find('title')
    if title_tag:
        title_text = title_tag.get_text()
        episode_number = html_file.stem.split('_')[0]
        episode_match = re.search(r'#(\d+):?\s*"?([^"]*)"?', title_text)
        if episode_match:
            episode_title = episode_match.group(2).strip()
        else:
            episode_title = title_text.replace(" - Buffy Episode Transcript", "").strip()
    else:
        episode_title = "Unknown"

    writer, airdate = "Unknown", "Unknown"

    writer_elements = soup.find_all(string=re.compile(f"Written by", re.IGNORECASE))
    if writer_elements:
        writer_line = writer_elements[0].strip()
        writer_match = re.search(r'Written by:?\s*(.+)', writer_line)
        if writer_match:
            writer = writer_match.group(1).strip()

    air_date_elements = soup.find_all(text=re.compile(f"Original Air Date (.+)", re.IGNORECASE))
    if air_date_elements:
        air_date_line = air_date_elements[0].strip()
        date_match =re.search(r'Original Air Date:?\s*(.+)', air_date_line)
        if date_match:
            airdate = date_match.group(1).strip()

    start_point = find_content_start(soup)
    if not start_point:
        return {'episode_num': episode_number, 'episode_title': episode_title,
                'writer': 'Unknown', 'air_date': 'Unknown', 'dialogue': []}

    dialogue_data = []

    if start_point.name == 'blockquote':
        dialogue_data = parse_blockquote_format(start_point, episode_number, episode_title)
    elif start_point.name == 'pre':
        dialogue_data = parse_pre_format(start_point, episode_number, episode_title)
    else:
        current_scene = "Unknown"
        for element in start_point.find_all_next(['b', 'p', 'br']):
            if element.name == 'p':
                # Handle <p> blocks that might contain multiple dialogue lines
                dialogue_lines = parse_dialogue_block(element, episode_number, episode_title)
                if dialogue_lines:
                    for line_data in dialogue_lines:
                        line_data['scene'] = current_scene
                        dialogue_data.append(line_data)
                        # print(
                        #     f"DEBUG: MATCH! Character: '{line_data['character']}', Dialogue: '{line_data['dialogue'][:50]}...'")
                else:
                    # No dialogue found, check if it's a single line dialogue (fallback)
                    text = clean_text(element.get_text())
                    if text:
                        dialogue_match = re.match(r'^([A-Za-z][A-Za-z\s]+?):\s*(.+)', text)

                        if dialogue_match:
                            character = dialogue_match.group(1).strip().upper()
                            if is_valid_character(character):
                                raw_dialogue = dialogue_match.group(2).strip()
                                dialogue = clean_dialogue(raw_dialogue)
                                dialogue_data.append({
                                    'episode_title': episode_title,
                                    'episode_num': episode_number,
                                    'scene': current_scene,
                                    'character': character,
                                    'dialogue': dialogue,
                                    'is_voiceover': 'VOICEOVER' in character.upper()
                                })
                                print(
                                    f"DEBUG: FALLBACK MATCH! Character: '{character}', Dialogue: '{dialogue[:50]}...'")

            elif element.name == 'b':
                # Scene markers
                text = clean_text(element.get_text())
                print(text)
                if text and len(text) < 50 and any(
                        word in text.lower() for word in ['teaser', 'act', 'scene', 'fade', 'cut']):
                    current_scene = text
                    # print(f"DEBUG: Scene change to: '{current_scene}'")

                if element.name == 'b' and not ':' in text:
                    if len(text) < 50 and any(word in text.lower() for word in ['teaser', 'act', 'scene', 'fade', 'cut']):
                        current_scene = text
                        continue

                dialogue_match = re.match(r'^([A-Za-z][A-Za-z\s]+?):\s*(.+)', text)
                if dialogue_match:
                    character = dialogue_match.group(1).strip()
                    if is_valid_character(character):
                        raw_dialogue = dialogue_match.group(2).strip()
                        dialogue = clean_dialogue(raw_dialogue)
                        if not dialogue:
                            continue

                        is_voiceover = 'VOICEOVER' in character.upper()

                        dialogue_data.append({
                            'episode_title': episode_title,
                            'episode_num': episode_number,
                            'scene': current_scene,
                            'character': character,
                            'dialogue': dialogue,
                            'is_voiceover': is_voiceover
                        })
                        print(f"DEBUG: MATCH! Character: '{character}', Dialogue: '{dialogue[:50]}...'")
                # print(f"DEBUG: NO MATCH for: '{text[:50]}...'")

    return {
        'episode_num': episode_number,
        'episode_title': episode_title,
        'writer': writer,
        'air_date': airdate,
        'dialogue': dialogue_data
    }

def parse_all_transcripts(html_files, sublist=None):
    all_episodes = []
    all_dialogues = []

    html_files = list(html_files.glob('*.html')) + list(html_files.glob('*.htm'))
    print(f"Found {len(html_files)} HTML files to process...")

    for html_file in sorted(html_files):
        if sublist and html_file.name not in sublist.values():
            continue
        print(f"Parsing {html_file.name}...")
        # try:
        episode_data = parse_buffy_transcript(html_file)
        all_episodes.append({
            'filename': html_file.name,
            'episode_num': episode_data['episode_num'],
            'episode_title': episode_data['episode_title'],
            'writer': episode_data['writer'],
            'air_date': episode_data['air_date'],
            'total_lines': len(episode_data['dialogue'])
        })

        for line in episode_data['dialogue']:
            line_data = line.copy()
            line_data.update({
                'episode_num': episode_data['episode_num'],
                'episode_title': episode_data['episode_title'],
                'filename': html_file.name
            })
            all_dialogues.append(line_data)
        # except Exception as e:
        #     print(f"Error processing {html_file.name}: {e}")
        #     continue

    episodes_df = pd.DataFrame(all_episodes)
    dialogues_df = pd.DataFrame(all_dialogues)

    print(f"\nParsing complete!")
    print(f"Episodes: {len(episodes_df)}")
    print(f"Total dialogue lines: {len(dialogues_df)}")

    return episodes_df, dialogues_df


def find_content_start(soup):
    """Find where the actual transcript content starts--order of if statements matters"""
    pre_tag = soup.find('pre')
    if pre_tag:
        print("Found <pre> tag - using as start point")
        return pre_tag

    blockquote = soup.find('blockquote')
    if blockquote:
        # Make sure it has actual dialogue content, not just metadata
        blockquote_text = blockquote.get_text()
        if len(blockquote_text) > 1000:  # Substantial content
            print("Using blockquote as start point (Season 6/7)")
            return blockquote

    # Look for <hr> tags
    hrs = soup.find_all('hr')
    if len(hrs) >= 2:
        # Try each <hr> tag and see which one has content after it
        for i, hr in enumerate(hrs):
            text_content = hr.find_all_next(['p', 'b'])
            if len(text_content) > 10:  # Arbitrary threshold
                print(f"Using <hr> tag {i} as start point ({len(text_content)} elements after)")
                return hr

    # Look for specific text patterns that indicate start of transcript
    transcript_markers = [
        'Teaser', 'TEASER', 'Act One', 'ACT ONE', 'Previously', 'PREVIOUSLY',
        'Episode begins', 'Episode opens', 'Open on', 'Fade in', "Prologue"
    ]

    for marker in transcript_markers:
        element = soup.find(text=re.compile(marker, re.IGNORECASE))
        if element:
            # Find the parent element
            parent = element.parent
            print(f"Found transcript marker '{marker}' in {parent.name} tag")
            return parent

    # Strategy 5: Look for the first <p> tag with substantial content
    p_tags = soup.find_all('p')
    for p in p_tags:
        text = clean_text(p.get_text())
        if len(text) > 50:  # Substantial content
            print(f"Using first substantial <p> tag: '{text[:50]}...'")
            return p

    print("ERROR: Could not find transcript start!")
    return None


def clean_data(episodes_df, dialogue_df):
    # Split up episodes by season
    bins = [0, 12, 34, 56, 78, 100, 122, 144]
    labels = [1, 2, 3, 4, 5, 6, 7]

    episodes_df['season'] = pd.cut(
        episodes_df['episode_num'].astype(int),
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    dialogue_df['season'] = pd.cut(
        dialogue_df['episode_num'].astype(int),
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    # Seasons 6 and 7 over-parse--instead of battling html formatting, just dedeupe
    print(f"Before deduplication: {len(dialogue_df)} lines")
    dialogue_df_clean = dialogue_df.drop_duplicates(subset=['episode_num', 'character', 'dialogue'])
    print(f"After deduplication: {len(dialogue_df_clean)} lines")
    dialogue_df_clean.to_csv('dialogue_cleaned.csv', index=False)

    new_episode_counts = dialogue_df.groupby('episode_num').size().reset_index(name='total_lines')

    # Update the episodes dataframe
    episodes_df = episodes_df.merge(
        new_episode_counts[['episode_num', 'total_lines']],
        on='episode_num',
        how='left',
        suffixes=('_old', '')
    )

    # Fill any missing counts with 0 (episodes with no dialogue parsed)
    episodes_df['total_lines'] = episodes_df['total_lines'].fillna(0).astype(int)

    # Drop the old column
    episodes_df = episodes_df.drop('total_lines_old', axis=1)

    # Recalculate seasons and check the results
    episodes_df['episode_num_int'] = episodes_df['episode_num'].astype(int)
    episodes_df['season'] = pd.cut(
        episodes_df['episode_num_int'],
        bins=[0, 12, 34, 56, 78, 99, 122, 144],
        labels=[1, 2, 3, 4, 5, 6, 7]
    )
    episodes_df.to_csv('episodes_cleaned.csv', index=False)


def explore_data():
    dialogue_df = pd.read_csv('all_dialogue.csv')
    episodes_df = pd.read_csv('episodes_cleaned.csv')


    #
    # print("\n=== AFTER CLEANING ===")
    # episodes_summary = dialogue_df_clean.groupby('episode_num').size().reset_index(name='total_lines')
    # episodes_summary['season'] = pd.cut(episodes_summary['episode_num'].astype(int),
    #                                     bins=[0, 12, 34, 56, 78, 99, 122, 144], labels=[1, 2, 3, 4, 5, 6, 7])
    #
    # print(episodes_summary.groupby('season')['total_lines'].agg(['count', 'mean']))
    #
    # low_line_episodes = episodes_df[episodes_df['total_lines'] < 1000]
    # subset_dialogue = dialogue_df[dialogue_df['episode_num'].isin(low_line_episodes['episode_num'])]
    #
    # print(f"Episodes with < 1000 lines: {len(low_line_episodes)}")
    # print(f"Dialogue lines from those episodes: {len(subset_dialogue)}")




    print("\n=== BY SEASON ===")
    print(episodes_df.groupby('season')['total_lines'].agg(['count', 'mean', 'min', 'max']))

    print("=== EPISODE OVERVIEW ===")
    print(episodes_df[['episode_title', 'total_lines']].sort_values('total_lines'))

    print("\n=== DIALOGUE SAMPLE ===")
    print(dialogue_df.head(10))

    print("\n=== CHARACTER FREQUENCY ===")
    print(dialogue_df['character'].value_counts().head(10))

    print("\n=== SCENES OVERVIEW ===")
    print(dialogue_df['scene'].value_counts().head())


if __name__ == "__main__":
    transcript_dir = Path(Path.cwd() / 'transcripts')
    episodes_df = pd.read_csv('episodes_cleaned.csv')
    dialogue_df = pd.read_csv('all_dialogue.csv')
    # stashed_dialogue_df = pd.read_csv('stash/dialogue_cleaned.csv')
    # season_6 = stashed_dialogue_df.loc[stashed_dialogue_df['season']==6].drop_duplicates()
    #
    # all_dialogue = pd.concat([season_6, dialogue_df], ignore_index=True)
    # # all_dialogue.sort_values(by='episode_num', inplace=True)
    # # all_dialogue.drop_duplicates(inplace=True)
    # clean_data(episodes_df, all_dialogue)
    # all_dialogue.to_csv('all_dialogue.csv', index=False)
    explore_data()

    # test_files = ["001_trans.html", "013_trans.html", "079_trans.html", "123_trans.html"]
    # for filename in transcript_dir.iterdir():
    #     if (Path(transcript_dir) / filename).exists():
    #         debug_single_file(filename)
    # Test one file from each problematic season
    test_files = {
        # 5: "099_trans.html",  # Or whatever season 1 files look like
        6: "108_trans.html",
        # 7: "140_trans.html"
    }

    # episodes_df, dialogues_df = parse_all_transcripts(transcript_dir)
    # clean_data(episodes_df, dialogues_df)
    # explore_data()

    # episodes_df.to_csv('episodes_new.csv', index=False)
    # dialogues_df.to_csv('dialogues_new.csv', index=False)




