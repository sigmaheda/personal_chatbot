import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    tokens = text.split()
    filtered = [stemmer.stem(token) for token in tokens if token.lower() not in stop_words]
    return ' '.join(filtered).lower()

def create_qa_pairs(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    qa_pairs, questions, answers = [], [], []

    regex = re.compile(r'\[\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}(?::\d{2})?\s?(AM|PM)?\]\s(.*?):\s(.*?)$')

    for i in range(len(lines) - 1):
        try:
            match_q = regex.search(lines[i])
            match_a = regex.search(lines[i + 1])

            if not (match_q and match_a):
                continue

            speaker1, question = match_q.group(2).strip(), match_q.group(3).strip()
            speaker2, answer = match_a.group(2).strip(), match_a.group(3).strip()

            if speaker1.lower() == speaker2.lower():
                continue
            if not question or not answer or '<Media omitted>' in question or '<Media omitted>' in answer:
                continue

            processed_q = preprocess(question)
            processed_a = preprocess(answer)

            qa_pairs.append((processed_q, processed_a))
            questions.append(processed_q)
            answers.append(processed_a)

        except Exception as e:
            print(f"Skipping due to error: {e}")
            continue

    return qa_pairs, questions, answers

def initialize_vectorizer(questions):
    vectorizer = TfidfVectorizer()
    question_vectors = vectorizer.fit_transform(questions)
    return vectorizer, question_vectors

def get_response(user_input, vectorizer, question_vectors, answers):
    user_input = preprocess(user_input)
    input_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(input_vector, question_vectors)
    best_match_index = similarities.argmax()

    if similarities[0, best_match_index] > 0.25:
        return answers[best_match_index]
    else:
        return "I don't understand. Try rephrasing?"

if __name__ == "__main__":
    filepath = 'chat.txt'  # Replace with your actual file
    qa_pairs, questions, answers = create_qa_pairs(filepath)

    if not qa_pairs:
        print("No valid Q&A pairs found. Check your file format.")
        exit()

    vectorizer, question_vectors = initialize_vectorizer(questions)

    print("Chatbot is ready! Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == 'exit':
            break
        response = get_response(user_input, vectorizer, question_vectors, answers)
        print("Chatbot:", response)
