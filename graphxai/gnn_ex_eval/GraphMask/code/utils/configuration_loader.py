import json


class ConfigurationLoader:

    def process_configuration(self, raw_configuration):
        return raw_configuration

    def load(self, filename):
        with open(filename, 'r') as json_file:
            json_data = json_file.read()
            configuration = json.loads(json_data)

        return self.process_configuration(configuration)