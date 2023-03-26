#Samantha Romero B87033
#Josef Ruzhicka B87095
from mimetypes import init


class Nodo():
    
    def __init__(self, value = None, node_type = None, gini = None,
                 count = None, split_column = None, split_type = None,
                 leftChild = None, rightChild = None, node_class = None, split_value = None):
        self.value        = value
        self.leftChild    = leftChild     
        self.rightChild   = rightChild   
        self.node_type    = node_type
        self.gini         = gini
        self.split_type   = split_type
        self.split_column = split_column
        self.split_value  = split_value
        self.node_class   = node_class
        self.count        = count
        pass 
    