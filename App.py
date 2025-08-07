from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Predefined FAQs
faqs = {
    "What is Artificial Intelligence?": "AI is the simulation of human intelligence by machines.",
    "What is machine learning?": "Machine learning is a subset of AI that learns from data.",
    "What is Python used for?": "Python is used for web development, data science, automation, and more.",
    "What is NLP?": "Natural Language Processing allows machines to understand and process human language.",
    "What is deep learning?": "Deep learning is a class of machine learning based on neural networks."
}

@app.route("/", methods=["GET", "POST"])
def chatbot():
    answer = ""
    user_question = ""
    if request.method == "POST":
        user_question = request.form["question"]
        answer = get_answer(user_question)
    return render_template("chat.html", question=user_question, answer=answer)

def get_answer(user_input):
    questions = list(faqs.keys())
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(questions + [user_input])
    similarity = cosine_similarity(vectors[-1], vectors[:-1])
    best_match = similarity.argmax()

    # Check similarity threshold (optional)
    if similarity[0][best_match] < 0.2:
        return "Sorry, I don’t understand the question. 🤖"
    return faqs[questions[best_match]]

if __name__ == "__main__":
    app.run(debug=True)
