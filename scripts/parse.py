from bs4 import BeautifulSoup
import pandas as pd
import re
from pathlib import Path
import openai
import json
import time
from typing import List, Dict

openai.api_key = "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"

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

    # Get all child elements in order
    all_elements = blockquote_element.find_all(['p', 'b', 'br'], recursive=False)

    i = 0
    while i < len(all_elements):
        element = all_elements[i]

        if element.name == 'b':
            text = clean_text(element.get_text())

            # Check if this looks like a character name
            if is_valid_character(text):
                character = text.upper()

                # Look for dialogue in the next <p> tag
                if i + 1 < len(all_elements) and all_elements[i + 1].name == 'p':
                    dialogue_text = clean_text(all_elements[i + 1].get_text())

                    # Skip if it's clearly stage direction
                    if (dialogue_text and
                            not dialogue_text.startswith('Cut to:') and
                            not dialogue_text.startswith('INT.') and
                            not dialogue_text.startswith('EXT.') and
                            len(dialogue_text) > 5):

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

                i += 2  # Skip both the <b> and following <p>
            else:
                # Might be a scene marker
                if any(word in text.lower() for word in ['ext.', 'int.', 'day', 'night']):
                    current_scene = text[:50]
                i += 1
        else:
            i += 1

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


def is_problematic_chunk(text: str) -> bool:
    """Identify chunks that need LLM cleaning"""
    problematic_indicators = [
        # Mixed dialogue and stage directions
        len(re.findall(r'[A-Z]+:', text)) > 1,  # Multiple speakers in one block
        # HTML artifacts
        '&' in text and ';' in text,  # HTML entities
        '<' in text or '>' in text,  # Leftover HTML tags
        # Transcriber notes
        'transcriber' in text.lower(),
        'note:' in text.lower(),
        # Stage directions mixed with dialogue
        '(' in text and ')' in text and ':' in text,
        # Very long single "dialogue" lines (probably mixed content)
        len(text) > 500,
    ]

    return any(problematic_indicators)


def llm_clean_chunk(text: str, episode_title: str) -> List[Dict]:
    """Use LLM to clean a problematic text chunk"""

    prompt = f"""Extract dialogue from this Buffy the Vampire Slayer transcript chunk.

Episode: {episode_title}

Rules:
1. Extract ONLY character dialogue. Character's name may be in all caps or capitalized, ie 'BUFFY: Hey Giles, what's up?' or 'Buffy: Hey Giles...'
2. Format as JSON array: [{{"character": "BUFFY", "dialogue": "Hey Giles, what's up?"}}, ...]
3. Remove stage directions, transcriber notes, HTML artifacts
4. Remove parentheticals from start of dialogue like "(to Xander)" 
5. Split multiple speakers into separate entries
6. Keep character names in ALL CAPS
7. If no dialogue found, return empty array []

Valid characters include: BUFFY, WILLOW, XANDER, GILES, ANGEL, SPIKE, CORDELIA, OZ, TARA, ANYA, RILEY, JOYCE, DAWN, FAITH, WESLEY, etc.

Text chunk:
{text}

Return only valid JSON array:"""

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1000
        )

        response_text = response.choices[0].message.content.strip()

        # Try to parse JSON response
        try:
            dialogue_list = json.loads(response_text)
            if isinstance(dialogue_list, list):
                return dialogue_list
        except json.JSONDecodeError:
            print(f"Warning: LLM returned invalid JSON: {response_text[:100]}...")

        return []  # Return empty if parsing fails

    except Exception as e:
        print(f"Error calling LLM: {e}")
        return []

def hybrid_parse_transcript(html_file):
    """Hybrid parsing: HTML first, then LLM for problematic chunks"""

    # Start with basic HTML parsing
    with open(html_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    soup = BeautifulSoup(content, 'html.parser')

    # Extract episode metadata (same as before)
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

    # Get raw text chunks
    start_point = find_content_start(soup)
    if not start_point:
        return {
            'episode_num': episode_number,
            'episode_title': episode_title,
            'writer': 'Unknown',
            'air_date': 'Unknown',
            'dialogue': []
        }

    # Collect all text chunks
    text_chunks = []
    current_scene = "Unknown"

    for element in start_point.find_all_next(['p', 'b']):
        raw_text = clean_text(element.get_text())

        if not raw_text:
            continue

        # Check if it's a scene marker
        if (element.name == 'b' and len(raw_text) < 50 and
                any(word in raw_text.lower() for word in ['teaser', 'act', 'scene', 'fade', 'cut'])):
            current_scene = raw_text
            continue

        text_chunks.append({
            'text': raw_text,
            'scene': current_scene,
            'is_problematic': is_problematic_chunk(raw_text)
        })

    # Process chunks
    dialogue_data = []
    llm_calls = 0
    max_llm_calls = 50  # Limit to control costs

    for chunk in text_chunks:
        if chunk['is_problematic'] and llm_calls < max_llm_calls:
            print(f"🤖 LLM cleaning: {chunk['text'][:50]}...")
            llm_calls += 1

            # Use LLM to clean
            llm_dialogue = llm_clean_chunk(chunk['text'], episode_title)

            for line in llm_dialogue:
                if 'character' in line and 'dialogue' in line:
                    character = line['character'].upper()
                    if is_valid_character(character):
                        dialogue_data.append({
                            'episode_num': episode_number,
                            'episode_title': episode_title,
                            'scene': chunk['scene'],
                            'character': character,
                            'dialogue': line['dialogue'],
                            'is_voiceover': 'VOICEOVER' in character
                        })

            # Rate limiting
            time.sleep(0.1)  # Small delay between API calls

        else:
            # Use regex parsing for clean chunks
            dialogue_match = re.match(r'^([A-Za-z][A-Za-z\s]+?):\s*(.+)', chunk['text'])
            if dialogue_match:
                character = dialogue_match.group(1).strip().upper()
                if is_valid_character(character):
                    raw_dialogue = dialogue_match.group(2).strip()
                    dialogue = clean_dialogue(raw_dialogue)

                    if dialogue:
                        dialogue_data.append({
                            'episode_num': episode_number,
                            'episode_title': episode_title,
                            'scene': chunk['scene'],
                            'character': character,
                            'dialogue': dialogue,
                            'is_voiceover': 'VOICEOVER' in character
                        })

    print(f"📊 Episode {episode_number}: {len(dialogue_data)} lines, {llm_calls} LLM calls")

    return {
        'episode_num': episode_number,
        'episode_title': episode_title,
        'writer': 'Unknown',  # You can add this back if needed
        'air_date': 'Unknown',
        'dialogue': dialogue_data
    }


def parse_all_transcripts_hybrid(transcript_dir, test_files=None):
    """Parse all transcripts using hybrid approach"""

    all_episodes = []
    all_dialogues = []

    html_files = list(Path(transcript_dir).glob("*.html"))

    if test_files:
        html_files = [f for f in html_files if f.name in test_files]

    print(f"🚀 Processing {len(html_files)} files with hybrid parsing...")

    total_llm_calls = 0

    for i, html_file in enumerate(html_files):
        print(f"\n📄 Processing {html_file.name} ({i + 1}/{len(html_files)})...")

        try:
            episode_data = hybrid_parse_transcript(html_file)

            # Add episode metadata
            all_episodes.append({
                'filename': html_file.name,
                'episode_num': episode_data['episode_num'],
                'episode_title': episode_data['episode_title'],
                'writer': episode_data['writer'],
                'air_date': episode_data['air_date'],
                'total_lines': len(episode_data['dialogue'])
            })

            # Add all dialogue
            all_dialogues.extend(episode_data['dialogue'])

        except Exception as e:
            print(f"❌ Error processing {html_file.name}: {e}")
            continue

    episodes_df = pd.DataFrame(all_episodes)
    dialogues_df = pd.DataFrame(all_dialogues)

    print(f"\n✅ Hybrid parsing complete!")
    print(f"📊 Episodes: {len(episodes_df)}")
    print(f"💬 Total dialogue lines: {len(dialogues_df)}")
    print(f"🤖 Estimated API cost: ~${total_llm_calls * 0.001:.2f}")  # Rough estimate

    return episodes_df, dialogues_df


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

def diagnose_parsing_issues(transcript_dir):
    """Debug what's going wrong with parsing"""
    html_files = list(Path(transcript_dir).glob("*.html"))

    for html_file in html_files[:5]:  # Test first 5 files
        print(f"\n=== DIAGNOSING {html_file.name} ===")

        with open(html_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        soup = BeautifulSoup(content, 'html.parser')

        # Check if we're finding the start point correctly
        hrs = soup.find_all('hr')
        print(f"Found {len(hrs)} <hr> tags")

        if len(hrs) >= 2:
            start_point = hrs[1]
            print("Using second <hr> as start point")
        else:
            start_point = soup.find('blockquote')
            print(f"Using blockquote as start point: {start_point is not None}")

        if start_point:
            # Count different element types after start point
            p_tags = start_point.find_all_next('p')
            b_tags = start_point.find_all_next('b')
            print(f"Found {len(p_tags)} <p> tags and {len(b_tags)} <b> tags after start point")

            # Sample some <p> tag content
            for i, p in enumerate(p_tags[:3]):
                text = clean_text(p.get_text())
                print(f"P tag {i}: '{text[:100]}...'")
        else:
            print("ERROR: No start point found!")

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

    writer_elements = soup.find_all(text=re.compile(f"Written by", re.IGNORECASE))
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
        # Handle pre formats
        dialogue_data = parse_pre_format(start_point, episode_number, episode_title)
    else:
        current_scene = "Unknown"
        for element in start_point.find_all_next(['b', 'p', 'br']):
            if element.name == 'p':
                # Handle <p> blocks that might contain multiple dialogue lines
                dialogue_lines = parse_dialogue_block(element, episode_number, episode_title)
                if dialogue_lines:
                    text_found = True
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
                if text and len(text) < 50 and any(
                        word in text.lower() for word in ['teaser', 'act', 'scene', 'fade', 'cut']):
                    text_found = True
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

    for html_file in html_files:
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
    """Find where the actual transcript content starts"""
    pre_tag = soup.find('pre')
    if pre_tag:
        print("Found <pre> tag - using as start point")
        return pre_tag

    # Strategy 2: Look for <hr> tags (existing logic)
    hrs = soup.find_all('hr')
    if len(hrs) >= 2:
        # Try each <hr> tag and see which one has content after it
        for i, hr in enumerate(hrs):
            test_content = hr.find_all_next(['p', 'b'])
            if len(test_content) > 10:  # Arbitrary threshold
                print(f"Using <hr> tag {i} as start point ({len(test_content)} elements after)")
                return hr

    # Strategy 3: Look for blockquote
    blockquote = soup.find('blockquote')
    if blockquote:
        print("Using blockquote as start point")
        return blockquote

    # Strategy 4: Look for specific text patterns that indicate start of transcript
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


def explore_data():
    """
    Quick data exploration to verify parsing worked correctly.
    """
    episodes_df = pd.read_csv('episodes.csv')
    dialogue_df = pd.read_csv('dialogues.csv')

    low_line_episodes = episodes_df[episodes_df['total_lines'] < 1000]
    subset_dialogue = dialogue_df[dialogue_df['episode_num'].isin(low_line_episodes['episode_num'])]

    print(f"Episodes with < 1000 lines: {len(low_line_episodes)}")
    print(f"Dialogue lines from those episodes: {len(subset_dialogue)}")

    # Save the subset
    subset_dialogue.to_csv('low_line_episodes_dialogue.csv', index=False)
    # Check if it's specific seasons or patterns
    bins = [0, 12, 34, 56, 78, 99, 122, 144]
    labels = [1, 2, 3, 4, 5, 6, 7]

    low_line_episodes['season'] = pd.cut(
        low_line_episodes['episode_num'].astype(int),
        bins=bins,
        labels=labels,
        include_lowest=True
    )
    print("\n=== BY SEASON ===")
    print(low_line_episodes.groupby('season')['total_lines'].agg(['count', 'mean']))

    # Let's see what's going on with these low-line episodes
    print("=== LOW LINE EPISODES ANALYSIS ===")
    print(low_line_episodes[['episode_title', 'total_lines']].sort_values('total_lines'))
    print("=== EPISODE OVERVIEW ===")
    print(episodes_df.head())

    print("\n=== DIALOGUE SAMPLE ===")
    print(dialogue_df.head(10))

    print("\n=== CHARACTER FREQUENCY ===")
    print(dialogue_df['character'].value_counts().head(10))

    print("\n=== SCENES OVERVIEW ===")
    print(dialogue_df['scene'].value_counts().head())


def debug_single_file(filename):
    filepath = Path(transcript_dir) / filename
    print(f"\n=== DEBUGGING {filename} ===")

    episode_data = parse_buffy_transcript(filepath)

    print(f"Episode: {episode_data['episode_title']}")
    print(f"Dialogue lines found: {len(episode_data['dialogue'])}")

    if len(episode_data['dialogue']) > 0:
        print("Sample dialogue:")
        for line in episode_data['dialogue'][:3]:
            print(f"  {line['character']}: {line['dialogue'][:50]}...")
    else:
        print("❌ NO DIALOGUE FOUND!")

    return episode_data


if __name__ == "__main__":
    transcript_dir = Path('/Users/ellynkeith/PycharmProjects/buffybot/transcripts')
    # test_files = ["001_trans.html", "013_trans.html", "079_trans.html", "123_trans.html"]
    # for filename in test_files:
    #     if (Path(transcript_dir) / filename).exists():
    #         debug_single_file(filename)
    # Test one file from each problematic season
    test_files = {
        1: "001_trans.html",  # Or whatever season 1 files look like
        2: "013_trans.html",
        5: "079_trans.html",
        7: "123_trans.html"
    }

    episodes_df, dialogues_df = parse_all_transcripts(transcript_dir, test_files)
    # episodes_df.to_csv('episodes_new.csv', index=False)
    # dialogues_df.to_csv('dialogues_new.csv', index=False)

    # for season, filename in test_files.items():
    #     if (Path(transcript_dir) / filename).exists():
    #         quick_format_check(filename)

    # Test with just a few problematic files first
    # test_files = ['043_trans.html'] # '068_trans.html', '129_trans.html',




