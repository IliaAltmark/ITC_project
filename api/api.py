import numpy as np
import pandas as pd
import tensorflow as tf
from model import get_model
from scipy.special import softmax
from flask import Flask, request, make_response, jsonify


app = Flask('Sentimental analysis')


@app.route('/get_sentiment')
def get_sentiment():
    if request.is_json:
        try:
            req = request.get_json()
            tokenized = tokenizer(req.get('text'))
            pred = model.predict({k: np.array(tokenized[k])[None] for k in input_names})[0]
            scores = softmax(pred, axis=1)

            return make_response(jsonify({'Sentiment': sentiments[np.argmax(scores)],
                                          'certain': round(float(np.max(scores)), 2)}), 200)
        except Exception as ex:
            print(ex)
            return make_response(jsonify({'error': str(ex)}), 500)
    else:
        return make_response(jsonify({"message": "No JSON"}), 400)


@app.route('/')
def test():
    return f'Use https://{request.host}/get_sentiment to get the result!!!'


if __name__ == '__main__':
    tokenizer, model = get_model()
    sentiments = ['Negative', 'Neutral', 'Positive']
    input_names = ['input_ids', 'token_type_ids', 'attention_mask']
    data_types = ({k: tf.int32 for k in input_names}, tf.int64)
    data_shapes = ({k: tf.TensorShape([None]) for k in input_names}, tf.TensorShape([]))

    app.run(host='0.0.0.0')
