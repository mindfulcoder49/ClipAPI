from fastapi import FastAPI
from flask import Flask

app = FastAPI()
flask_app = Flask(__name__)

# You can have both apps running side by side if needed
