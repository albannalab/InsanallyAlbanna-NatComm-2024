from abc import ABC, abstractmethod 

class NetworkVisualization(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def init_figure(self):
        pass

    @abstractmethod
    def update_figure(self):
        pass

    @abstractmethod
    def figure_properties(self):
        pass
    

    