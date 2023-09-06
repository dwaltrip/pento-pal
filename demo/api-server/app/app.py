from flask import Flask, request
app = Flask(__name__)

from PIL import Image
from pento_parser_poc import parse_puzzle_solution


@app.route('/test-endpoint')
def test_endpoint():
    return {
        'test-json': 'test-json-value'
    }


@app.route('/parse-solution', methods=['POST'])
def parse_solution():
    image = request.files.get('image', None)

    if not image:
        return dict(error='no image provided')

    image = Image.open(image)
    # image.show()

    return dict(
        result=parse_puzzle_solution(image)
    )