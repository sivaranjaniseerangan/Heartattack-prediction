import pickle


def heartattack_prediction(features):
    pickled_model = pickle.load(open('model_prediction.pkl', 'rb'))
    target = str(round(list(pickled_model.predict([features]))[0]))

    return str("Heart Attack " + target)

test_features=[51.0, 0.0, 0.0, 130.0, 305.0, 0.0, 1.0, 142.0, 1.0, 1.2, 1.0, 0.0, 3.0]
heartattack_prediction(test_features)