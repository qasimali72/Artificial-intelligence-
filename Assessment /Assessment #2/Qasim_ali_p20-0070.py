import math
import re
from collections import defaultdict

# Load stopwords
stop_words = [
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and',
    'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being',
    'below', 'between', 'both', 'but', 'by', "can't", 'cannot', 'could', "couldn't",
    'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during',
    'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't",
    'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here',
    "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i',
    "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's",
    'its', 'itself', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself',
    'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought',
    'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she',
    "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than',
    'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then',
    'there', "there's", 'these', 'they', "they'd", "they'll", "they're", "they've",
    'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was',
    "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what',
    "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who',
    "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't", 'you',
    "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself',
    'yourselves'
]

# Clean and tokenize text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return [word for word in text.split() if word not in stop_words]

# Load and process data from TSV file
def load_data(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if '\t' in line:
                genre, desc = line.strip().split('\t', 1)
                data.append((genre.strip(), clean_text(desc)))
    return data

# Training and testing
train_data = load_data('film-genres-train.tsv')
test_data = load_data('film-genres-test.tsv')

# Train Naive Bayes
class_word_counts = defaultdict(lambda: defaultdict(int))
class_counts = defaultdict(int)
vocab = set()

for genre, words in train_data:
    class_counts[genre] += 1
    for word in words:
        class_word_counts[genre][word] += 1
        vocab.add(word)

# Predict function
def predict(words):
    scores = {}
    total_docs = sum(class_counts.values())
    for genre in class_counts:
        log_prob = math.log(class_counts[genre] / total_docs)
        total_words = sum(class_word_counts[genre].values())
        for word in words:
            count = class_word_counts[genre].get(word, 0) + 1
            log_prob += math.log(count / (total_words + len(vocab)))
        scores[genre] = log_prob
    return max(scores, key=scores.get)

# Evaluate
correct = defaultdict(int)
incorrect = defaultdict(int)

for actual, words in test_data:
    predicted = predict(words)
    if predicted == actual:
        correct[actual] += 1
    else:
        incorrect[actual] += 1

# Report
genres = set([g for g, _ in test_data])
for g in genres:
    print(f"{g} - Correct: {correct[g]}, Incorrect: {incorrect[g]}")
