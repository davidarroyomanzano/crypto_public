#save model
def save_model(model_name):
    import json
    from fbprophet.serialize import model_to_json, model_from_json
    with open(model_name, 'w') as fout:
        json.dump(model_to_json(m), fout)  # Save model
    with open(model_name, 'r') as fin:
        m = model_from_json(json.load(fin))  # Load model
