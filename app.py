from flask import Flask, render_template, request
import tradestation as ts
import pandas as pd

app = Flask(__name__)

@app.route("/")
def index():
    mode = request.args.get('mode', None)
    if mode:
        try:
            data = ts.main
        except:
            pass

    return render_template('index.html',data=data,mode=mode)
    