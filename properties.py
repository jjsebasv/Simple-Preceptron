import configparser
import json

class Properties:
	
	def __init__(self, filename):
		configParser = configparser.RawConfigParser()
		configParser.read(filename)
		self.etha = float(configParser.get('Algorithm', 'etha'))
		self.error = float(configParser.get('Algorithm', 'error'))
		self.filename = configParser.get('Pattern File', 'name')
		self.hidden_layer_sizes = json.loads(configParser.get('Hidden Layers', 'sizes'))