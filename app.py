from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import os
import xgboost
from text_preprocess import preprocess_text
from service import dataload
app = Flask(__name__)
CORS(app)

folder_path=os.getcwd()

#Define a route for your API endpoint and return a response
@app.route('/api')
def api():
    return 'This is your Flask API'

@app.route('/api/predictions/',methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        print('data',data)
        text = data.get('text')
        extraction_technique = data.get('featureExtractionTechnique')
        classifier = data.get('classifier')
        
        #apply preprocessing techniques on test data same as used in training
        text = preprocess_text(text)

        list_dataset1 = dataload(dataset_name='frd', 
                                 extraction_technique=extraction_technique, 
                                 classifier=classifier,
                                 text=text)
        list_dataset2 = dataload(dataset_name='ylp', 
                                 extraction_technique=extraction_technique, 
                                 classifier=classifier,
                                 text=text)
        resultData ={
             'dataset1':(np.array(list_dataset1)).tolist(),
             'dataset2':(np.array(list_dataset2)).tolist()
        }

        response = jsonify(resultData)
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', '*')
        response.headers.add('Access-Control-Allow-Methods', '*')

        return response
    
    except Exception as e:
            return jsonify({'error': str(e)})


#Run the app
if __name__ == '__main__':
    app.run(debug=True)