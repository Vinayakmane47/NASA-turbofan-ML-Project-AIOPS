
from flask import Flask , request 
import sys 

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def welcome(): 
    if (request.method == 'GET'): 
        return "Welcome to Turbofan Project"
        

if __name__ == "__main__": 
    app.run()
