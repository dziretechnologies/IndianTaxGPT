from langchain import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
from constants import CHROMA_SETTINGS, PERSIST_DIRECTORY
import os
from constants import CHROMA_SETTINGS
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import subprocess
from flask_pymongo import PyMongo
import json
from bson import json_util

# Here we are setting chatbot
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
device = 'cpu'
embeddings = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-xl", model_kwargs={"device": device})
# load the vectorstore
db = Chroma(persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
retriever = db.as_retriever()
repo_id = "tiiuae/falcon-7b-instruct"
llm = HuggingFaceHub(repo_id=repo_id,
                     model_kwargs={"temperature": 0.6, "max_new_tokens": 500})
qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)


# Here we are setting Flask Server
app = Flask(__name__)
CORS(app)
app.config['MONGO_URI'] = 'mongodb://localhost:27017/indiantaxgpt'
mongo = PyMongo(app)
if mongo:
    print("DB Connected")
else:
    print("Database could not connect")


@app.route('/login', methods=['POST'])
def login():
    x_token = request.json.get('x_token')
    email = request.json.get('email')
    name = request.json.get('name')
    profile_pic = request.json.get('profile_pic')
    social_id = request.json.get('social_id')
    login_type = request.json.get('login_type')

    # Create or access the 'users' collection in the 'indiantaxgpt' database
    users_collection = mongo.db.users

    # Update the document in the collection
    update = {
        '$set': {
            'email': email,
            'name': name,
            'x_token': x_token,
            'profile_pic': profile_pic,
            'social_id': social_id,
            'login_type': login_type
        }
    }
    user = users_collection.find_one_and_update(
        {'email': email}, update, upsert=True, return_document=True)

    # Convert the document to JSON
    user_dict = json.loads(json_util.dumps(user))
    user_dict['_id'] = str(user_dict['_id']['$oid'])
    return jsonify({"status": True, "code": 200, "message": "Login successful", "info": user_dict})


# Here we are setting end points or we can say apis
@app.route('/query')
def index():
    authorization = request.headers.get('Authorization')
    exists = mongo.db.users.find_one({'x_token':authorization})
    if exists:
        query = request.args.get('question')
        res = qa(query)
        answer, docs = res['result'], res['source_documents']
        return jsonify({"status": True, "code": 200, "answer": answer})
    else:
        return jsonify({"status":False,"code":401,"message":"Session expired, please login again"})


@app.route('/upload', methods=['POST'])
def upload_files():
    uploaded_files = request.files.getlist('files')
    for file in uploaded_files:
        # You can perform operations on each file
        file.save('SOURCE_DOCUMENTS/' + file.filename)
    subprocess.run(['python3', 'ingest.py', '--device_type', 'cpu'])
    return jsonify({"status": True, "code": 200})


if __name__ == "__main__":
    app.run()
