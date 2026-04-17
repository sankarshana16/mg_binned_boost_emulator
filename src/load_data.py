import numpy as np





class Standardizer():
    def __init__(self, base_array):

      self.mean = base_array.mean(axis=0)  
      self.std = base_array.std(axis=0)
    
    def standardize(self,array):
        
      standardized_array = (array-self.mean)/self.std

      return standardized_array

    def destandardize(self,standardized_array):

      array = standardized_array*self.std + self.mean

      return array
