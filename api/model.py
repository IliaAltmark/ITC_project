from transformers import BertTokenizer
from tensorflow.keras.callbacks import EarlyStopping
from transformers import TFBertForSequenceClassification
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy


def get_model():
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    model.load_weights('saved_weights/BERT_sentiment_model_alejandro')

    sparse_categorical_crossentropy = SparseCategoricalCrossentropy(
        from_logits=True, name="sparse_categorical_crossentropy"
    )

    optimizer = Adam(
        learning_rate=1e-5, decay=1e-6
    )

    es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    # model.compile(optimizer=optimizer, loss=sparse_categorical_crossentropy, metrics=['accuracy'])

    return BertTokenizer.from_pretrained('bert-base-uncased'), model

