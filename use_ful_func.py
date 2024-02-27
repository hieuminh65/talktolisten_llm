import re

def remove_special_characters(s):
    """Remove special characters from a string"""
    return re.sub(r'[^A-Za-z0-9 ]+', '', s)

def get_topic(file):
    with open(f'{file}', 'r') as f:
        lines = f.readlines()
    topics = lines
    topic_list = []
    for topic in topics:
        if "###" in topic:
            continue
        else:
            topic_list.append(topic.strip())
    return topic_list
