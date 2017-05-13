import configparser
import json

class Properties:

	def __init__(self, filename):
		configParser = configparser.RawConfigParser()
		configParser.read(filename)
		self.error = float(configParser.get('Algorithm', 'error'))
		self.error_freq = float(configParser.get('Algorithm', 'error_freq'))
		self.function_type = configParser.get('Backpropagation', 'type')
		self.beta = float(configParser.get('Backpropagation', 'function_beta'))
		self.etha = float(configParser.get('Backpropagation', 'etha'))
		self.use_non_zero_dg = configParser.get('Backpropagation', 'non_zero_dg') == "true"
		self.use_momentum = configParser.get('Backpropagation', 'momentum') == "true"
		self.use_adap_etha = configParser.get('Backpropagation', 'adap_etha') == "true"
		self.momentum_alpha = float(configParser.get('Backpropagation', 'momentum_alpha'))
		self.undo_probability = float(configParser.get('Backpropagation', 'undo_probability'))
		self.etha_a = float(configParser.get('Backpropagation', 'etha_a'))
		self.etha_b = float(configParser.get('Backpropagation', 'etha_b'))
		self.epoch_freq = float(configParser.get('Backpropagation', 'epoch_freq'))
		self.filename = configParser.get('Pattern File', 'name')
		self.training_file = configParser.get('Pattern File', 'training')
		self.test_file = configParser.get('Pattern File', 'test')
		self.training_percentage = float(configParser.get('Pattern File', 'training_percentage'))
		self.hidden_layer_sizes = json.loads(configParser.get('Hidden Layers', 'sizes'))
