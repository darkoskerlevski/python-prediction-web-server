from flask import Flask, request

app = Flask(__name__)


@app.route("/")
def home():
    return "Hello, World!"


@app.route('/web', methods=['GET', 'POST'])
def result():
    print(request.form['key','value']) # should display 'bar'
    return 'Received !' # response to your request.


if __name__ == "__main__":
    app.run(debug=True)