from enum import Enum

class Update_T(Enum):
  FULL = 1
  PARTIAL = 2
  RPARTIAL = 3
  MASKED = 4
  HALF = 5
  WEIGHTED = 6

# enum for random vector generator type
class Generator(Enum):
  Vanilla = 1

config = {

  "D" : 200, 
  "vector" : "Gaussian",  
  "mu" : 0,
  "sigma" : 1,
  "binarize" : 0, 
  "lr": 0.05, 
  "sparse" : 0, 
  "s" : 0.1,
  "binaryModel" : 0, 
  "checkpoints": False, 


  "alpha": 0.5,
  "beta": 1,
  "theta": 0.25,
  "regen_rate": 0.04
}
