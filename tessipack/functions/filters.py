import importlib

class Filters:
    def __init__(self):
        self.filters = {}
        #print('new filter')
        module_name = 'tessipack.functions.default_filters'
        self.load_filters_from_module(module_name)
    def add_filter(self, name, func):
        self.filters[name] = func

    def load_filters_from_module(self, module_name):
        module = importlib.import_module(module_name)
        for name in dir(module):
            obj = getattr(module, name)
            if callable(obj) :
                self.add_filter(name, obj)

    def load_filters_from_file(self, file_path):
        # Create a dictionary to store loaded functions
        loaded_functions = {}
        # Load the file's code using exec
        with open(file_path, 'r') as file:
            code = compile(file.read(), file_path, 'exec')
            exec(code, loaded_functions)

        for name, obj in loaded_functions.items():
            if callable(obj) :
                self.add_filter(name, obj)


    def apply_filter(self, name=None, data=None,**kwargs):
        if name in self.filters:
            filter_function = self.filters[name]
            filtered_data = filter_function(data=data,**kwargs)
            return filtered_data
        else:
            raise ValueError(f"Filter '{name}' not found.")


# # Create an instance of the Filters class
# filters = Filters()

# # Load filters from a module
# module_name = 'filter_functions'
# filters.load_filters_from_module(module_name)

# # Load filters from a file
# file_path = 'custom_filters.py'
# filters.load_filters_from_file(file_path)

# # Example usage
# data = ...  # Your data to be filtered
# filter_name = 'filter_function_1'

# if filter_name in filters.filters:
#     filter_function = filters.filters[filter_name]
#     filtered_data = filter_function(data)
#     print("Filtered data:", filtered_data)
# else:
#     print(f"Filter '{filter_name}' not found.")
