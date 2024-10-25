from flask import Flask, render_template, request, flash
from flask_sqlalchemy import SQLAlchemy
import pickle
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import os


# Download NLTK stopwords (only needed for the first run)
nltk.download('stopwords')

# Prepare stopwords and the Porter Stemmer
stopwords_set = set(stopwords.words('english'))
porter = PorterStemmer()
emoticon_pattern = re.compile(r'(?::|;|=)(7:-)?(?:\)|\(|D|P)')

# Initialize the Flask app
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///sentiment_history.db '
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize the database
db = SQLAlchemy(app)

# Define the CommentHistory model for saving comments and sentiments
class CommentHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    comment = db.Column(db.Text, nullable=False)
    sentiment = db.Column(db.String(50), nullable=False)

# Load the sentiment analysis model and TF-IDF vectorizer
with open('clf.pkl', 'rb') as f:
    clf = pickle.load(f)
with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

def preprocessing(text):
    # Remove any HTML tags
    text = re.sub('<[^>]*>', '', text)

    # Change all letters to lowercase and remove special symbols
    text = re.sub(r'\W+', ' ', text.lower())

    # Split the text into words
    words = text.split()

    # Handle negations for words like "not", "no", "never"
    for i in range(len(words)):
        if words[i] in ["not", "no", "never"]:
            if i + 1 < len(words):
                words[i + 1] = f"NOT_{words[i + 1]}"

    # Remove stopwords and stem the words
    words = [porter.stem(word) for word in words if word not in stopwords_set]

    return " ".join(words)

@app.route('/', methods=['GET', 'POST'])
def analyze_sentiment():
    if request.method == 'POST':
        comment = request.form.get('comment')

        # Preprocess the comment
        preprocessed_comment = preprocessing(comment)

        # Transform the preprocessed comment into a feature vector
        comment_vector = tfidf.transform([preprocessed_comment])

        # Predict the sentiment
        sentiment = clf.predict(comment_vector)[0]

        # Save the comment and sentiment to the database
        history_entry = CommentHistory(comment=comment, sentiment=sentiment)
        db.session.add(history_entry)
        db.session.commit()


        return render_template('index.html', sentiment=sentiment)

    return render_template('index.html')

@app.route('/history', methods=['GET'])
def view_history():
    # Retrieve all entries from CommentHistory
    history = CommentHistory.query.all()
    return render_template('history.html', history=history)


# Run the app
if __name__ == '__main__':
    # Create the database file if it doesn't already exist
    if not os.path.exists("sentiment_history.db"):
        with app.app_context():
            db.create_all()
    app.run(debug=True)
