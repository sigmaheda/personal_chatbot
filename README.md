# Chatbot from Personal Messages

A simple chatbot built using TF-IDF and cosine similarity, trained directly on your personal message history (e.g., WhatsApp export). The bot mimics your conversational style based on past Q&A pairs.

## 🔧 Features
- Extracts real Q&A pairs from chat logs
- Preprocesses text with stemming and stopword removal
- Uses TF-IDF to find the most relevant answer
- Lightweight, no deep learning required

## 📂 Structure
- `chatbot.py`: Main script to run the chatbot
- `data/chat.txt`: Your WhatsApp export (not included in repo) or any other chat history in text format
- `requirements.txt`: Python dependencies

## 🚀 Setup

1. Clone the repo:
```bash
git clone https://github.com/your-username/chatbot-from-messages.git
cd chatbot-from-messages
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Add your chat export file in the `data/` folder.

4. Run the chatbot:
```bash
python chatbot.py
```

## 💬 Format Assumption
The bot expects chat lines in the WhatsApp format:
```
[12/03/23, 10:15:00 AM] John: Hello
[12/03/23, 10:15:05 AM] Priya: Hi!
```

## 📝 License
This project is licensed under the [MIT License](./LICENSE).
