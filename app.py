from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load trained model
with open("recommender_model.pkl", "rb") as f:
    model_data = pickle.load(f)

vectorizer = model_data["vectorizer"]
clf = model_data["model"]
job_labels = model_data["job_labels"]

def recommend_roles(resume_text, top_n=3):
    vec = vectorizer.transform([resume_text])
    probs = clf.predict_proba(vec)[0]

    # Get top roles
    top_indices = probs.argsort()[-top_n:][::-1]
    top_roles = [(job_labels[i], probs[i]) for i in top_indices]

    # Scale scores to 100%
    total = sum(p for _, p in top_roles)
    scaled = [(role, round((p / total) * 100, 2)) for role, p in top_roles]

    return scaled

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        resume_text = request.form["resume"]
        recommendations = recommend_roles(resume_text)
        return render_template("result.html", roles=recommendations)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
