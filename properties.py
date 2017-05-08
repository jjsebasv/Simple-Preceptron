import configparser
import json

class Properties:

	def __init__(self, filename):
		configParser = configparser.RawConfigParser()
		configParser.read(filename)
		self.etha = float(configParser.get('Algorithm', 'etha'))
		self.error = float(configParser.get('Algorithm', 'error'))
		self.filename = configParser.get('Pattern File', 'name')
		self.training_file = configParser.get('Pattern File', 'training')
		self.test_file = configParser.get('Pattern File', 'test')
		self.training_percentage = float(configParser.get('Pattern File', 'training_percentage'))
		self.hidden_layer_sizes = json.loads(configParser.get('Hidden Layers', 'sizes'))
