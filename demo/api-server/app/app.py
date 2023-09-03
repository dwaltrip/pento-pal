from flask import Flask
app = Flask(__name__)


@app.route('/test-endpoint')
def test_endpoint():
    return {
        'test-json': 'test-json-value'
    }
